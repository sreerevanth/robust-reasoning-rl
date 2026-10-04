"""Optional real optimizer test: random tiny local GPT-2, no network or benchmark claims."""

import copy
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
    cfg["model"] = {"backend": "huggingface", "name": str(model_dir)}
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
    payload = train(cfg)
    assert payload["metadata"]["execution_kind"] == "training_complete"
    assert payload["reward_statistics"]["samples"] >= 2
    assert (Path(payload["checkpoint"]) / "adapter_config.json").exists()
    assert (tmp_path / "training/training.json").exists()
    from src.models.generation import HFGenerator

    inference = HFGenerator({**cfg["model"], "adapter": payload["checkpoint"]}, cfg["generation"])
    assert len(inference.generate("What is 2 + 3?", 2, 42)) == 2
    inference.config = {**inference.config, "temperature": 0}
    greedy = inference.generate("What is 2 + 3?", 2, 42)
    assert greedy[0] == greedy[1]
