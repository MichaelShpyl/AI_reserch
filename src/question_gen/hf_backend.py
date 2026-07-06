"""Local HuggingFace generation backend, with an optional QLoRA adapter.

Lets the question-generation pipeline run a transformers model directly (rather than via Ollama), so
a QLoRA-fine-tuned local model can be compared against its own base model on the same task. Same
`chat_json(system, user)` interface as the other backends. Kept separate from the Ollama path so the
main pipeline is unaffected.

Only loaded on demand by the fine-tuning evaluation; not imported by the default pipeline.
"""

from __future__ import annotations

import json
import re


def _clean_question(q: str) -> str:
    """Keep a single well-formed question: strip quotes/markers and cut at the first '?', which
    drops trailing JSON tails or extra fragments the fine-tuned model sometimes appends after the
    real question (e.g. `What is X? {"answer": ...}`)."""
    q = re.sub(r"^\s*[-*\d.)\"']+\s*", "", (q or "").strip()).strip()
    m = re.match(r"(.{5,}?\?)", q)
    return (m.group(1) if m else q).strip().strip('"').strip()


def _extract_json(text: str) -> dict:
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
        t = re.sub(r"\n?```$", "", t).strip()
    raw = []
    try:
        obj = json.loads(t)
        if isinstance(obj, dict) and isinstance(obj.get("questions"), list):
            raw = [str(x) for x in obj["questions"]]
    except Exception:
        pass
    if not raw:
        i, j = t.find("{"), t.rfind("}")
        if i != -1 and j != -1 and j > i:
            try:
                obj = json.loads(t[i:j + 1])
                if isinstance(obj.get("questions"), list):
                    raw = [str(x) for x in obj["questions"]]
            except Exception:
                pass
    if not raw:
        # Plain-text fallback: drop any leaked JSON objects first (so their key/value text cannot be
        # mistaken for a question), then pull each question segment up to its '?'.
        stripped = t
        for _ in range(3):
            stripped = re.sub(r"\{[^{}]*\}", " ", stripped)
        raw = re.findall(r"([A-Za-z][^?{}\[\]]{4,}?\?)", stripped)
    qs = []
    for r in raw:
        cq = _clean_question(r)
        if cq.endswith("?") and len(cq.split()) >= 3 and cq not in qs:
            qs.append(cq)
    return {"questions": qs[:5]}


class HFBackend:
    """A transformers causal-LM backend. `adapter` is an optional PEFT/LoRA directory."""

    def __init__(self, base_model: str = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
                 adapter: str | None = None, temperature: float = 0.2, max_new_tokens: int = 320):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        self.torch = torch
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.name = f"local-hf:{'finetuned' if adapter else 'base'}:{base_model.split('/')[-1]}"
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=torch.bfloat16,
                                   bnb_4bit_use_double_quant=True)
        self.tok = AutoTokenizer.from_pretrained(base_model)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model, quantization_config=quant, device_map={"": 0}, torch_dtype=torch.bfloat16)
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()

    def chat_json(self, system: str, user: str) -> dict:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = self.tok(prompt, return_tensors="pt").to("cuda")
        with self.torch.no_grad():
            out = self.model.generate(
                **enc, max_new_tokens=self.max_new_tokens, do_sample=self.temperature > 0,
                temperature=max(self.temperature, 0.01), top_p=0.9,
                pad_token_id=self.tok.pad_token_id)
        text = self.tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return _extract_json(text)

    def close(self):
        del self.model
        self.torch.cuda.empty_cache()
