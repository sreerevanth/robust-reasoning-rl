"""Baseline and robust training differ only in reward shaping and output location."""

from pathlib import Path
from typing import Any

from src.data.loading import load_dataset
from src.models.generation import prompt_for
from src.rewards.factory import build_ensemble
from src.rewards.shaping import RewardConfig
from src.training.rewards import RewardFunction
from src.utils.config import validate
from src.utils.logging import event
from src.utils.persistence import provenance, write_json
from src.utils.reproducibility import seed_everything


def train(config: dict[str, Any], resume: str | None = None) -> dict[str, Any]:
    validate(config)
    if config["model"]["backend"] != "huggingface":
        raise ValueError("Training requires the huggingface backend; fixture training is not research")
    generations = config["generation"]["num_generations"]
    t = config["training"]
    if generations < 2 or t["batch_size"] * t["gradient_accumulation_steps"] % generations:
        raise ValueError("GRPO requires >=2 generations dividing effective per-device batch size")
    if config["generation"]["temperature"] <= 0:
        raise ValueError("GRPO training requires positive sampling temperature")
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    seed_everything(config["seed"])
    examples = load_dataset(config["dataset"])
    dataset = Dataset.from_list([
        {"prompt": prompt_for(e.question), "reference": e.reference, "example_id": e.id,
         "question": e.question, "example_metadata": e.metadata} for e in examples
    ])
    model_cfg = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg.get("tokenizer", model_cfg["name"]), revision=model_cfg.get("revision"),
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"], revision=model_cfg.get("revision"), trust_remote_code=False,
    )
    peft = None
    if t["lora"]["enabled"]:
        from peft import LoraConfig

        lora = t["lora"]
        peft = LoraConfig(r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
                          target_modules=lora["target_modules"], task_type="CAUSAL_LM")
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "run.json", provenance(config, execution_kind="training_started"))
    reward = RewardFunction(build_ensemble(config["training_verifier"]), RewardConfig(**config["reward"]))
    args = GRPOConfig(
        output_dir=str(output), learning_rate=t["learning_rate"],
        per_device_train_batch_size=t["batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        max_steps=t["max_steps"], save_steps=t["save_steps"], logging_steps=t["logging_steps"],
        num_generations=generations, max_completion_length=config["generation"]["max_tokens"],
        temperature=config["generation"]["temperature"], top_p=config["generation"]["top_p"],
        beta=t["beta"], seed=config["seed"], data_seed=config["seed"],
        gradient_checkpointing=t["gradient_checkpointing"], bf16=t["bf16"], use_cpu=t["use_cpu"],
        report_to="none", remove_unused_columns=False, scale_rewards=False,
    )
    event("training_started", model=model_cfg["name"], output=str(output), strategy=config["reward"])
    trainer = GRPOTrainer(model=model, args=args, train_dataset=dataset,
                          reward_funcs=reward, processing_class=tokenizer, peft_config=peft)
    try:
        result = trainer.train(resume_from_checkpoint=resume)
        trainer.save_model(str(output / "final"))
        tokenizer.save_pretrained(str(output / "final"))
        payload = {"metadata": provenance(config, execution_kind="training_complete"),
                   "metrics": result.metrics, "reward_statistics": dict(reward.statistics),
                   "trainer_log_history": trainer.state.log_history,
                   "checkpoint": str(output / "final")}
        if trainer.is_world_process_zero():
            write_json(output / "training.json", payload)
        return payload
    except Exception as error:
        if trainer.is_world_process_zero():
            write_json(output / "failure.json", {"type": type(error).__name__, "message": str(error)})
        raise
