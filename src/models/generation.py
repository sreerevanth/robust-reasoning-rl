"""Fixture backend is a software test, never a simulated trained language model."""

import hashlib
import random
import re
from fractions import Fraction
from typing import Any, Protocol


def prompt_for(question: str) -> str:
    return f"Solve the problem. End with 'Final answer: <answer>'.\nProblem: {question}\nSolution:"


def policy_prompt(question: str, model_config: dict[str, Any]) -> str | list[dict[str, str]]:
    """Match TRL conversational rendering and inference for instruction-tuned models."""
    text = prompt_for(question)
    return (
        [{"role": "user", "content": text}] if model_config.get("prompt_format") == "chat" else text
    )


class Generator(Protocol):
    def generate(self, question: str, count: int, seed: int) -> list[str]: ...


class FixtureGenerator:
    """Parses fixture arithmetic from questions; deliberately makes seeded mistakes."""

    def generate(self, question: str, count: int, seed: int) -> list[str]:
        match = re.fullmatch(r"What is (\d+) ([+*/-]) (\d+)\?", question)
        if not match:
            raise ValueError("Fixture backend supports only tiny arithmetic questions")
        left, operator, right = match.groups()
        a, b = Fraction(left), Fraction(right)
        value = {"+": lambda: a + b, "-": lambda: a - b, "*": lambda: a * b, "/": lambda: a / b}[
            operator
        ]()
        key = int.from_bytes(hashlib.sha256(question.encode()).digest()[:8], "big")
        rng = random.Random(seed ^ key)
        return [
            f"Fixture arithmetic. Final answer: {value if rng.random() < 0.7 else value + 1}"
            for _ in range(count)
        ]


class HFGenerator:
    def __init__(self, model_config: dict[str, Any], generation_config: dict[str, Any]):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.config = generation_config
        self.model_config = model_config
        self.last_generation_metadata: list[dict[str, Any]] = []
        if model_config.get("cpu_threads"):
            torch.set_num_threads(model_config["cpu_threads"])
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_config.get("tokenizer", model_config["name"]),
            revision=model_config.get("revision"),
            trust_remote_code=False,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_config["name"],
            revision=model_config.get("revision"),
            trust_remote_code=False,
            **({"dtype": getattr(torch, model_config["dtype"])} if "dtype" in model_config else {}),
        )
        if model_config.get("adapter"):
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, model_config["adapter"])
        device = model_config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, question: str, count: int, seed: int) -> list[str]:
        import torch
        from transformers import set_seed

        set_seed(seed)
        prompt = policy_prompt(question, self.model_config)
        if isinstance(prompt, list):
            prompt = self.tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True
            )
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(
            self.model.device
        )
        options = {
            "max_new_tokens": self.config["max_tokens"],
            "pad_token_id": self.tokenizer.pad_token_id,
            "do_sample": self.config["temperature"] > 0,
        }
        if options["do_sample"]:
            options.update(temperature=self.config["temperature"], top_p=self.config["top_p"])
            if "top_k" in self.config:
                options["top_k"] = self.config["top_k"]
        outputs = []
        self.last_generation_metadata = []
        eos_ids = self.model.generation_config.eos_token_id
        if eos_ids is None:
            eos_ids = self.tokenizer.eos_token_id
        eos_ids = eos_ids if isinstance(eos_ids, list) else [eos_ids]
        eos_ids = [token for token in eos_ids if token is not None]
        # One call per sample keeps deterministic decoding compatible with count > 1.
        with torch.inference_mode():
            batch_size = self.config.get("sample_batch_size", 1) if options["do_sample"] else 1
            for start in range(0, count, batch_size):
                tokens = self.model.generate(
                    **inputs, **options, num_return_sequences=min(batch_size, count - start)
                )
                for sequence in tokens:
                    generated = sequence[inputs["input_ids"].shape[1] :]
                    # Batched generation pads sequences after EOS. Exclude this padding.
                    eos_positions = torch.isin(
                        generated, torch.tensor(eos_ids, device=generated.device)
                    ).nonzero()
                    terminated = len(eos_positions) > 0
                    if terminated:
                        generated = generated[: int(eos_positions[0].item()) + 1]
                    self.last_generation_metadata.append(
                        {
                            "generated_tokens": len(generated),
                            "prompt_tokens": inputs["input_ids"].shape[1],
                            "finish_reason": "eos" if terminated else "length",
                            "device": str(self.model.device),
                            "eos_token_ids": eos_ids,
                            "do_sample": options["do_sample"],
                        }
                    )
                    outputs.append(self.tokenizer.decode(generated, skip_special_tokens=True))
        return outputs


class RecordedGenerator:
    """Replay actual base responses across corruption levels without rerunning inference."""

    def __init__(self, records: list[dict[str, Any]]):
        self.records = records
        self.last_generation_metadata: list[dict[str, Any]] = []

    def generate(self, question: str, count: int, seed: int) -> list[str]:
        matches = [record for record in self.records if record["metadata"]["question"] == question]
        if len(matches) != count:
            raise ValueError("Recorded base generation budget mismatch")
        self.last_generation_metadata = [
            {**record["metadata"], "reused_base_generation": True} for record in matches
        ]
        return [record["response"] for record in matches]


def build_generator(config: dict[str, Any]) -> Generator:
    if config["model"]["backend"] == "fixture":
        return FixtureGenerator()
    return HFGenerator(config["model"], config["generation"])
