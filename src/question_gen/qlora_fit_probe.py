"""QLoRA fit probe: does an 8B-class model fine-tune inside the 8 GB RTX 4060?

This answers the Backend B question flagged at Meeting 4. It loads Llama 3.1 8B pre-quantized to
4-bit NF4 (the unsloth bnb-4bit mirror, so no gated-repo token and no 16 GB download), attaches a
LoRA adapter, runs real training steps at increasing sequence lengths with gradient checkpointing
and a paged 8-bit optimiser, and records peak VRAM per configuration. The output JSON is the
evidence for the 8B-vs-fallback decision.

    python src/question_gen/qlora_fit_probe.py            # 8B probe
    python src/question_gen/qlora_fit_probe.py --model Qwen/Qwen2.5-3B-Instruct   # fallback probe
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "qlora_fit_probe.json"


def gb(x: int) -> float:
    return round(x / (1024 ** 3), 2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Meta-Llama-3.1-8B-bnb-4bit",
                    help="HF model id; default is the pre-quantized 4-bit Llama 3.1 8B mirror")
    ap.add_argument("--seq-lens", type=int, nargs="+", default=[256, 512, 1024])
    args = ap.parse_args()

    import torch
    import bitsandbytes as bnb
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    assert torch.cuda.is_available(), "CUDA GPU required for the probe"
    dev = torch.cuda.get_device_properties(0)
    report = {
        "model": args.model,
        "gpu": dev.name,
        "vram_total_gb": gb(dev.total_memory),
        "config": "4-bit NF4, LoRA r=16 on q/k/v/o, grad checkpointing, PagedAdamW8bit, batch 1",
        "steps": [],
    }
    print(f"GPU: {dev.name} ({gb(dev.total_memory)} GB)", flush=True)
    print(f"Loading {args.model} in 4-bit ...", flush=True)
    t0 = time.time()

    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.bfloat16,
                               bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=quant, device_map={"": 0},
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    report["load_seconds"] = round(time.time() - t0, 1)
    report["vram_after_load_gb"] = gb(torch.cuda.memory_allocated())
    print(f"Loaded in {report['load_seconds']}s; VRAM after load {report['vram_after_load_gb']} GB",
          flush=True)

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    model = get_peft_model(model, lora)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    report["trainable_params_millions"] = round(trainable / 1e6, 1)
    print(f"LoRA attached: {report['trainable_params_millions']}M trainable params", flush=True)

    opt = bnb.optim.PagedAdamW8bit((p for p in model.parameters() if p.requires_grad), lr=2e-4)
    model.train()
    base = ("Write one verification question for this claim from a student essay. "
            "Claim: language shapes how people understand the world. ") * 200

    for seq in args.seq_lens:
        entry = {"seq_len": seq}
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            enc = tok(base, return_tensors="pt", truncation=True, max_length=seq).to("cuda")
            enc["labels"] = enc["input_ids"].clone()
            t1 = time.time()
            out = model(**enc)
            out.loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            entry.update({
                "ok": True,
                "step_seconds": round(time.time() - t1, 2),
                "loss": round(float(out.loss), 4),
                "peak_vram_allocated_gb": gb(torch.cuda.max_memory_allocated()),
                "peak_vram_reserved_gb": gb(torch.cuda.max_memory_reserved()),
            })
            print(f"seq {seq}: OK, step {entry['step_seconds']}s, "
                  f"peak alloc {entry['peak_vram_allocated_gb']} GB "
                  f"(reserved {entry['peak_vram_reserved_gb']} GB)", flush=True)
        except torch.cuda.OutOfMemoryError:
            entry.update({"ok": False, "error": "CUDA out of memory"})
            print(f"seq {seq}: OOM", flush=True)
            torch.cuda.empty_cache()
        except Exception as e:
            entry.update({"ok": False, "error": f"{type(e).__name__}: {e}"})
            print(f"seq {seq}: FAILED {e}\n{traceback.format_exc()}", flush=True)
        report["steps"].append(entry)

    fits = [s["seq_len"] for s in report["steps"] if s.get("ok")]
    report["verdict"] = (
        f"Training steps completed at seq lens {fits} on {gb(dev.total_memory)} GB VRAM"
        if fits else "No configuration fitted; use the smaller fallback model")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    # Merge with earlier probes so 8B and fallback results live side by side.
    existing = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {"probes": []}
    existing["probes"] = [p for p in existing.get("probes", []) if p.get("model") != args.model]
    existing["probes"].append(report)
    OUT.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(f"\nVERDICT: {report['verdict']}")
    print(f"Saved {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
