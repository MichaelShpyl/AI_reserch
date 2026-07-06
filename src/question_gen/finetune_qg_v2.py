"""Re-fine-tune the local backend on OPEN-ENDED questions (Backend B experiment, v2).

The v1 fine-tune (finetune_qg.py) trained on EduQG and collapsed into degenerate multiple-choice stems
(qg_quality_audit.py: 95% "Which of the following..."), because 47% of EduQG's own questions are
multiple-choice. This version changes ONLY the training data, to SQuAD, whose questions are open-ended
and passage-grounded, and keeps every other setting identical to v1 so the comparison isolates the
effect of the data format. If v2's questions are well-formed where v1's were degenerate, the diagnosis
"the data format dominates" is confirmed.

The adapter is saved to a NEW directory (`models/qg_finetune_qwen3b_v2/`); the failed v1 adapter is
kept for the record. QLoRA: 4-bit NF4, LoRA on the attention projections, gradient checkpointing,
paged 8-bit optimiser, batch 1, prompt tokens masked, two epochs over 2,600 pairs.

    python src/question_gen/finetune_qg_v2.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
ADAPTER_DIR = REPO / "models" / "qg_finetune_qwen3b_v2"
BASE_MODEL = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
SEED = 42
MAX_LEN = 512
N_EXAMPLES = 2600

# Identical training prompt to v1, so only the data differs.
SYSTEM = ("You are a teacher. Read the passage and write one clear question that checks whether a "
          "student understood it. Reply with the question only.")


def load_pairs() -> list[dict]:
    """Open-ended passage->question pairs from SQuAD. One question per context for topic diversity,
    filtered to substantive open questions (>= 6 words, ends with '?'). No multiple-choice anywhere."""
    from datasets import load_dataset
    ds = load_dataset("squad", split="train")
    seen, pairs = set(), []
    for r in ds:
        q = r["question"].strip()
        ctx = " ".join(r["context"].split())
        key = r["context"][:100]
        if key in seen:
            continue
        if not q.endswith("?") or len(q.split()) < 6 or len(ctx) < 80:
            continue
        # guard against any multiple-choice phrasing leaking in (there is none in SQuAD, but be safe)
        if "which of the following" in q.lower():
            continue
        seen.add(key)
        pairs.append({"passage": ctx[:1500], "question": q})
    return pairs


def main() -> int:
    import torch
    from datasets import Dataset
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                              Trainer, TrainingArguments, set_seed)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    set_seed(SEED)
    assert torch.cuda.is_available(), "CUDA required"
    pairs = load_pairs()
    rng = np.random.default_rng(SEED)
    rng.shuffle(pairs)
    pairs = pairs[:N_EXAMPLES]
    mcq = sum(1 for p in pairs if "which of the following" in p["question"].lower())
    print(f"{len(pairs)} open-ended QG examples from SQuAD (multiple-choice in sample: {mcq})", flush=True)
    print("  e.g.:", pairs[0]["question"], flush=True)

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def build(ex):
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
    (REPO / "outputs" / "qg_finetune_v2.json").write_text(json.dumps({
        "base_model": BASE_MODEL, "n_examples": len(pairs), "data": "SQuAD context->question (open-ended)",
        "config": "QLoRA r=16 q/k/v/o, 2 epochs, bs1 x grad-accum 8, paged AdamW 8-bit (identical to v1)",
        "final_loss": round(float(trainer.state.log_history[-1].get("train_loss", 0)), 4),
        "note": "v2: same config as v1, only the training data changed (EduQG -> SQuAD) to fix the "
                "multiple-choice degeneracy. Audit the output for degeneracy before trusting any score.",
    }, indent=2), encoding="utf-8")
    print(f"Saved adapter to {ADAPTER_DIR.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
