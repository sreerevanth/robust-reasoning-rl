"""Optional real optimizer test: random tiny local GPT-2, no network or benchmark claims."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.training.runner import train
from src.utils.config import DEFAULTS


@pytest.mark.parametrize("strategy", ["standard", "combined"])
def test_real_local_grpo(tmp_path, strategy):
    torch = pytest.importorskip("torch")
    pytest.importorskip("trl")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    torch.set_num_threads(1)
    torch.manual_seed(1)
    vocab = {
        word: i
        for i, word in enumerate(
            ["[PAD]", "[EOS]", "[UNK]", "Final", "answer", ":", "1", "2", "3", "4", "5", "."]
        )
    }
    tok = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tok, pad_token="[PAD]", eos_token="[EOS]", unk_token="[UNK]"
    )
    model_dir = tmp_path / "model"
    tokenizer.save_pretrained(model_dir)
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=len(vocab),
            n_layer=1,
            n_head=1,
            n_embd=16,
            n_positions=128,
            bos_token_id=1,
            eos_token_id=1,
            pad_token_id=0,
        )
    )
    model.save_pretrained(model_dir)
    cfg = copy.deepcopy(DEFAULTS)
    cfg["model"] = {"backend": "huggingface", "name": str(model_dir), "cpu_threads": 1}
    cfg["dataset"]["path"] = str(Path(__file__).resolve().parents[1] / "data/tiny_train.jsonl")
    cfg["generation"].update(num_generations=2, max_tokens=8)
    cfg["evaluation"]["k"] = [1, 2]
    cfg["training"].update(
        use_cpu=True,
        batch_size=2,
        max_steps=1,
        logging_steps=1,
        save_steps=1,
        gradient_checkpointing=False,
    )
    cfg["training"]["lora"].update(enabled=True, target_modules=["c_attn"])
    cfg["reward"] = {"strategy": strategy}
    cfg["output_dir"] = str(tmp_path / "training")
    if strategy == "standard":
        import yaml

        config_path = tmp_path / "cli-training.yaml"
        config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        subprocess.run(
            [sys.executable, "-m", "src.cli", "train", "--config", str(config_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads((tmp_path / "training/training.json").read_text())
    else:
        payload = train(cfg)
    assert payload["metadata"]["execution_kind"] == "training_complete"
    assert payload["reward_statistics"]["samples"] >= 2
    assert payload["policy_movement"]["trainable_parameter_count"] > 0
    assert payload["policy_movement"]["optimizer_steps"] == 1
    assert payload["policy_movement"]["initial_parameter_hashes"]
    assert (Path(payload["checkpoint"]) / "adapter_config.json").exists()
    assert (tmp_path / "training/training.json").exists()
    from src.models.generation import HFGenerator

    inference = HFGenerator({**cfg["model"], "adapter": payload["checkpoint"]}, cfg["generation"])
    assert len(inference.generate("What is 2 + 3?", 2, 42)) == 2
    inference.config = {**inference.config, "sample_batch_size": 2}
    assert len(inference.generate("What is 2 + 3?", 2, 42)) == 2
    assert len(inference.last_generation_metadata) == 2
    assert all(1 <= m["generated_tokens"] <= 8 for m in inference.last_generation_metadata)
    inference.config = {**inference.config, "temperature": 0}
    greedy = inference.generate("What is 2 + 3?", 2, 42)
    assert greedy[0] == greedy[1]
