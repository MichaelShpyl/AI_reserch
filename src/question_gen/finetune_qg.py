"""Preliminary QLoRA fine-tune of a local model for question generation (Backend B experiment).

PRELIMINARY, pending supervisor sign-off on the 8B-to-3B scope change. The fit probes
(outputs/qlora_fit_probe.json) showed the locked-scope 8B trains only by spilling into system RAM
(214 s/step), while Qwen2.5 3B trains comfortably, so this experiment uses the 3B to produce evidence
for that decision, not to pre-empt it. It does NOT alter the locked-scope document.

The model is fine-tuned to generate an educational question from a passage, using the EduQG corpus
(Hadifar et al. 2022): the highlighted answer-bearing sentences are the passage, the question is the
target. Prompt tokens are masked so the model learns only to produce the question. QLoRA: 4-bit NF4,
LoRA on the attention projections, gradient checkpointing, paged 8-bit optimiser, batch 1.

    python src/question_gen/finetune_qg.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
EDUQG = REPO / "data" / "raw" / "eduqg"
ADAPTER_DIR = REPO / "models" / "qg_finetune_qwen3b"
BASE_MODEL = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
SEED = 42
MAX_LEN = 512

SYSTEM = ("You are a teacher. Read the passage and write one clear question that checks whether a "
          "student understood it. Reply with the question only.")


def load_pairs() -> list[dict]:
    pairs = []
    for f in ["qg_train_v0.json", "qg_valid_v0.json"]:
        for ch in json.loads((EDUQG / f).read_text(encoding="utf-8")):
            for q in ch.get("questions", []):
                question = q["question"].get("normal_format")
                passage = q.get("hl_sentences") or q.get("hl_context") or ""
                passage = " ".join(passage.split())
                if question and len(passage) > 40:
                    pairs.append({"passage": passage[:1500], "question": question.strip()})
    return pairs


def main() -> int:
    import torch
    import bitsandbytes as bnb
    from datasets import Dataset
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                              Trainer, TrainingArguments, set_seed)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    set_seed(SEED)
    assert torch.cuda.is_available(), "CUDA required"
    pairs = load_pairs()
    rng = np.random.default_rng(SEED)
    rng.shuffle(pairs)
    pairs = pairs[:2600]
    print(f"{len(pairs)} QG examples from EduQG", flush=True)

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def build(ex):
        # Prompt (system + user + generation cue) is masked; the question is the target.
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Passage:\n{ex['passage']}"}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        full = prompt + ex["question"] + tok.eos_token
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        f_ids = tok(full, add_special_tokens=False, truncation=True, max_length=MAX_LEN)["input_ids"]
        labels = [-100] * len(p_ids) + f_ids[len(p_ids):]
        labels = labels[:len(f_ids)]
        return {"input_ids": f_ids, "attention_mask": [1] * len(f_ids), "labels": labels}

    ds = Dataset.from_list([build(p) for p in pairs])

    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=quant,
                                                 device_map={"": 0}, torch_dtype=torch.bfloat16)
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    model.config.use_cache = False
    print(f"trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.1f}M",
          flush=True)

    class PadCollator:
        def __call__(self, feats):
            m = max(len(f["input_ids"]) for f in feats)
            pad = tok.pad_token_id
            out = {"input_ids": [], "attention_mask": [], "labels": []}
            for f in feats:
                n = m - len(f["input_ids"])
                out["input_ids"].append(f["input_ids"] + [pad] * n)
                out["attention_mask"].append(f["attention_mask"] + [0] * n)
                out["labels"].append(f["labels"] + [-100] * n)
            return {k: torch.tensor(v) for k, v in out.items()}

    args = TrainingArguments(
        output_dir=str(ADAPTER_DIR / "_runs"), seed=SEED,
        num_train_epochs=2, per_device_train_batch_size=1, gradient_accumulation_steps=8,
        learning_rate=2e-4, warmup_ratio=0.03, weight_decay=0.0, lr_scheduler_type="cosine",
        logging_steps=25, save_strategy="no", report_to=[], bf16=True,
        optim="paged_adamw_8bit", max_grad_norm=0.3,
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=PadCollator())
    trainer.train()

    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ADAPTER_DIR))
    tok.save_pretrained(str(ADAPTER_DIR))
    (REPO / "outputs" / "qg_finetune.json").write_text(json.dumps({
        "base_model": BASE_MODEL, "n_examples": len(pairs), "data": "EduQG context->question",
        "config": "QLoRA r=16 q/k/v/o, 2 epochs, bs1 x grad-accum 8, paged AdamW 8-bit",
        "final_loss": round(float(trainer.state.log_history[-1].get("train_loss", 0)), 4),
        "note": "Preliminary Backend B experiment on Qwen2.5 3B; pending 8B-to-3B scope sign-off.",
    }, indent=2), encoding="utf-8")
    print(f"Saved adapter to {ADAPTER_DIR.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
