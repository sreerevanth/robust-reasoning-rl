"""Fixture backend is a software test, never a simulated trained language model."""

import hashlib
import random
import re
from fractions import Fraction
from typing import Any, Protocol


def prompt_for(question: str) -> str:
    return f"Solve the problem. End with 'Final answer: <answer>'.\nProblem: {question}\nSolution:"


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
        value = {"+": lambda: a + b, "-": lambda: a - b,
                 "*": lambda: a * b, "/": lambda: a / b}[operator]()
        key = int.from_bytes(hashlib.sha256(question.encode()).digest()[:8], "big")
        rng = random.Random(seed ^ key)
        return [f"Fixture arithmetic computation. Final answer: {value if rng.random() < .7 else value + 1}"
                for _ in range(count)]


class HFGenerator:
    def __init__(self, model_config: dict[str, Any], generation_config: dict[str, Any]):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.config = generation_config
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_config.get("tokenizer", model_config["name"]),
            revision=model_config.get("revision"), trust_remote_code=False,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_config["name"], revision=model_config.get("revision"),
            trust_remote_code=False,
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
        inputs = self.tokenizer(prompt_for(question), return_tensors="pt").to(self.model.device)
        options = {"max_new_tokens": self.config["max_tokens"],
                   "pad_token_id": self.tokenizer.pad_token_id,
                   "do_sample": self.config["temperature"] > 0}
        if options["do_sample"]:
            options.update(temperature=self.config["temperature"], top_p=self.config["top_p"])
        outputs = []
        # One call per sample keeps deterministic decoding compatible with count > 1.
        with torch.inference_mode():
            for _ in range(count):
                tokens = self.model.generate(**inputs, **options)
                outputs.append(self.tokenizer.decode(
                    tokens[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
        return outputs


def build_generator(config: dict[str, Any]) -> Generator:
    if config["model"]["backend"] == "fixture":
        return FixtureGenerator()
    return HFGenerator(config["model"], config["generation"])
