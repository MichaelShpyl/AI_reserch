"""Inference for the trained claim extractor: argument spans in the student's own words.

The training script (train_claim_extractor.py) fits a DeBERTa BIO tagger over MajorClaim / Claim /
Premise on paragraph-level sequences. This is the matching inference side: given an essay it returns
the argument spans the model finds, each with its type, its verbatim text, and its character offsets,
decoded from the BIO labels exactly as they were encoded during training (paragraph sequences, 256
subwords, offset mapping). It runs on CPU, which is enough for one essay and keeps the GPU free.

    from extract_spans import ClaimExtractor
    spans = ClaimExtractor().extract(text)   # [{type, text, start, end}, ...]
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MODELDIR = REPO / "models" / "claim_extractor"


def paragraphs_with_offsets(text: str):
    pos = 0
    for line in text.split("\n"):
        if line.strip():
            yield line, pos
        pos += len(line) + 1


class ClaimExtractor:
    def __init__(self, model_dir: Path = MODELDIR):
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForTokenClassification.from_pretrained(str(model_dir)).eval()
        self.id2label = self.model.config.id2label

    def _decode_paragraph(self, para: str, base: int):
        enc = self.tok(para, truncation=True, max_length=256, return_offsets_mapping=True,
                       return_tensors="pt")
        offsets = enc.pop("offset_mapping")[0].tolist()
        with self.torch.no_grad():
            logits = self.model(**enc).logits[0]
        labels = [self.id2label[int(i)] for i in logits.argmax(-1)]

        spans, cur = [], None
        for (a, b), lab in zip(offsets, labels):
            if a == b:                              # special token
                continue
            if lab.startswith("B-"):
                if cur:
                    spans.append(cur)
                cur = {"type": lab[2:], "start": base + a, "end": base + b}
            elif lab.startswith("I-") and cur and cur["type"] == lab[2:]:
                cur["end"] = base + b
            else:                                   # O, or a role switch without a B
                if cur:
                    spans.append(cur)
                cur = None
        if cur:
            spans.append(cur)
        return spans

    def extract(self, text: str) -> list[dict]:
        out = []
        for para, p0 in paragraphs_with_offsets(text):
            for s in self._decode_paragraph(para, p0):
                s["text"] = text[s["start"]:s["end"]].strip()
                if len(s["text"]) >= 3:
                    out.append(s)
        return out
