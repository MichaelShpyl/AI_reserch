"""The pipeline behind the interactive app, with the models loaded once instead of per request.

The batch scripts each load what they need and exit, which is fine for an experiment and useless
for a UI: hybrid_detect() alone reloads DeBERTa, spaCy, GPT-2, the gradient-boosting model and the
fuser on every call, which is several seconds a time. This module keeps one copy of each model in
process and reuses it, so the first request pays the loading cost and the rest do not.

Nothing here re-implements a model. Every function calls the same trained artefacts the
dissertation reports on, so the app and the results chapters cannot disagree.

    from pipeline_service import detect, explain, claims_and_questions, warm_up
"""

from __future__ import annotations

import json
import pickle
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HYBRID = REPO / "models" / "hybrid"
DETECTOR = REPO / "models" / "detector"

for sub in ("detection", "explainability", "question_gen"):
    p = str(REPO / "src" / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

MAXLEN = 512
_LOCK = threading.Lock()
_M: dict = {}          # the one and only model cache


# ----------------------------------------------------------------- model loading

def _load_all():
    """Load every local model once. Called under the lock, so it runs at most one time."""
    if _M.get("ready"):
        return
    t0 = time.time()
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import glob

    cps = glob.glob(str(DETECTOR / "checkpoint-*"))
    ckpt = max(cps, key=lambda p: int(p.split("-")[-1])) if cps else str(DETECTOR)
    _M["torch"] = torch
    _M["tok"] = AutoTokenizer.from_pretrained(ckpt)
    _M["deberta"] = AutoModelForSequenceClassification.from_pretrained(ckpt).eval()

    from text_normalize import normalize_text
    _M["normalize"] = normalize_text

    # The stylometric half. spaCy is the slow one to load, so it matters that this happens once.
    from stylometric import load_nlp, stylometric_features
    _M["nlp"] = load_nlp()
    _M["stylometric_features"] = stylometric_features

    if (HYBRID / "fuser.pkl").exists():
        from hybrid_fusion import gpt2_perplexity
        _M["gpt2_perplexity"] = gpt2_perplexity
        _M["feat_cols"] = json.loads((HYBRID / "feat_cols.json").read_text(encoding="utf-8"))
        _M["gbm"] = pickle.load(open(HYBRID / "gbm_ppl.pkl", "rb"))
        _M["fuser"] = pickle.load(open(HYBRID / "fuser.pkl", "rb"))
        _M["has_hybrid"] = True
    else:
        _M["has_hybrid"] = False

    _M["ready"] = True
    _M["load_seconds"] = round(time.time() - t0, 1)


def warm_up() -> dict:
    """Load the models up front so the first user request is not the one that waits."""
    with _LOCK:
        _load_all()
    return {"ready": True, "hybrid": _M["has_hybrid"], "load_seconds": _M["load_seconds"]}


def _ensure():
    if not _M.get("ready"):
        with _LOCK:
            _load_all()


# ----------------------------------------------------------------- detection

def _transformer_prob(norm_text: str) -> float:
    torch = _M["torch"]
    enc = _M["tok"](norm_text, truncation=True, max_length=MAXLEN, return_tensors="pt")
    with torch.no_grad():
        return float(torch.softmax(_M["deberta"](**enc).logits, -1)[0, 1])


def detect(text: str) -> dict:
    """Hybrid probability plus the two component views, exactly as Section 6.7 defines them."""
    _ensure()
    t0 = time.time()
    norm = _M["normalize"](text)
    p_deb = _transformer_prob(norm)

    if not _M["has_hybrid"]:
        return {"prob_ai": round(p_deb, 4), "flagged": p_deb >= 0.5, "detector": "transformer",
                "components": {"transformer": round(p_deb, 4)},
                "seconds": round(time.time() - t0, 2)}

    import numpy as np
    import pandas as pd
    feats = _M["stylometric_features"](norm, _M["nlp"])
    feats["gpt2_ppl"] = float(_M["gpt2_perplexity"]([norm], _M["torch"].device("cpu"))[0])
    X = pd.DataFrame([feats])[_M["feat_cols"] + ["gpt2_ppl"]]
    p_style = float(_M["gbm"].predict_proba(X)[0, 1])
    p_hy = float(_M["fuser"].predict_proba(np.array([[p_deb, p_style]]))[0, 1])
    return {"prob_ai": round(p_hy, 4), "flagged": p_hy >= 0.5, "detector": "hybrid",
            "components": {"transformer": round(p_deb, 4),
                           "style_plus_perplexity": round(p_style, 4)},
            "seconds": round(time.time() - t0, 2)}


# ----------------------------------------------------------------- explanation

def explain(text: str, top_k: int = 5) -> dict:
    """The habit card: each feature with its SHAP push and where it sits against real students.

    Returns {"rows": [...], "sentences": [...]}. The underlying function names the list "features"
    and also writes a PNG to disk; the app wants neither, so the shape is normalised here rather
    than leaving the front end to know about it.
    """
    _ensure()
    import tempfile
    from explain_submission import explain_submission
    with tempfile.TemporaryDirectory() as td:
        card = explain_submission(text, Path(td) / "card.png", top_k=top_k)
    return {"rows": card.get("features", []),
            "sentences": card.get("sentences", [])}


# ----------------------------------------------------------------- sentence evidence

# Phrases a large model over-uses relative to student writing. Counted, never assumed: the app
# reports how often each appears in THIS text so the reader can judge the evidence.
TELL_PHRASES = [
    "meanwhile", "furthermore", "moreover", "in contrast", "in conclusion", "additionally",
    "highlights the", "showcases", "underscores", "delves", "it is important to note",
    "plays a significant role", "a testament to", "quintessential", "nuanced",
    "multidimensional", "rich tapestry", "thought-provoking", "respectively", "overall",
]


def _logodds_batch(texts: list[str]):
    import numpy as np
    torch = _M["torch"]
    out = []
    for i in range(0, len(texts), 8):
        enc = _M["tok"](texts[i:i + 8], truncation=True, max_length=MAXLEN,
                        padding=True, return_tensors="pt")
        with torch.no_grad():
            lg = _M["deberta"](**enc).logits
        out.extend((lg[:, 1] - lg[:, 0]).tolist())
    return np.array(out)


def sentence_marks(text: str, top_k: int = 3) -> dict:
    """Which sentences the detector reacts to, by deleting each one and watching the score move.

    Reported in log-odds because the in-domain detector saturates near probability 1.0, where
    single-sentence removals vanish below rounding (Section 5.8). Each sentence also carries the
    evidence a reader needs to judge the mark for themselves: its length against the essay's own
    median, and any over-used phrases it contains, counted across the whole submission.
    """
    _ensure()
    import re
    import numpy as np
    import statistics
    from sentence_occlusion import split_sentences

    norm = _M["normalize"](text)
    sents = split_sentences(norm)
    if len(sents) < 2:
        return {"sentences": sents, "drops": [0.0] * len(sents), "top": [], "detail": []}

    full = float(_logodds_batch([norm])[0])
    variants = [" ".join(sents[:i] + sents[i + 1:]) for i in range(len(sents))]
    drops = full - _logodds_batch(variants)
    order = list(np.argsort(drops)[::-1])
    rank = {int(idx): r + 1 for r, idx in enumerate(order)}

    words = [len(re.findall(r"[A-Za-z']+", s)) for s in sents]
    median_len = statistics.median(words) if words else 0
    whole = norm.lower()
    counts = {p: len(re.findall(r"\b" + p.replace(" ", r"\s+") + r"\b", whole)) for p in TELL_PHRASES}
    counts = {p: c for p, c in counts.items() if c}

    detail = []
    for i, s in enumerate(sents):
        low = s.lower()
        hits = [{"phrase": p, "in_text": counts[p]} for p in counts
                if re.search(r"\b" + p.replace(" ", r"\s+") + r"\b", low)]
        detail.append({"index": i, "rank": rank[i], "drop": round(float(drops[i]), 4),
                       "words": words[i], "phrases": hits})

    return {"sentences": sents,
            "drops": [round(float(d), 4) for d in drops],
            "full_logodds": round(full, 3),
            "median_words": median_len,
            "phrase_counts": counts,
            "detail": detail,
            "top": [{"index": int(i), "drop": round(float(drops[i]), 4)} for i in order[:top_k]]}


def counterfactual(text: str, k: int = 3) -> dict:
    """What actually happens to the verdict if the most-reacted-to sentences are deleted.

    This is the question a lecturer asks next, and the answer is usually not the one they expect:
    on the essays measured for this project the score barely moves, because the machine-likeness is
    spread across the whole submission rather than sitting in a few sentences. Running it live on
    the submission in front of them is more convincing than being told.
    """
    _ensure()
    import numpy as np
    from sentence_occlusion import split_sentences

    norm = _M["normalize"](text)
    sents = split_sentences(norm)
    if len(sents) <= k + 1:
        return {"available": False}

    full = float(_logodds_batch([norm])[0])
    variants = [" ".join(sents[:i] + sents[i + 1:]) for i in range(len(sents))]
    drops = full - _logodds_batch(variants)
    top = list(np.argsort(drops)[::-1][:k])

    without_top = " ".join(s for i, s in enumerate(sents) if i not in top)
    lo_top = float(_logodds_batch([without_top])[0])

    rng = np.random.default_rng(42)          # fixed seed: the comparison must be reproducible
    rand_los = []
    for _ in range(5):
        idx = set(rng.choice(len(sents), size=k, replace=False).tolist())
        rand_los.append(float(_logodds_batch([" ".join(s for i, s in enumerate(sents) if i not in idx)])[0]))

    sig = lambda z: 1 / (1 + np.exp(-z))
    return {"available": True, "k": k,
            "full_logodds": round(full, 3),
            "after_top_logodds": round(lo_top, 3),
            "after_random_logodds": round(float(np.mean(rand_los)), 3),
            "prob_before": round(float(sig(full)), 4),
            "prob_after_top": round(float(sig(lo_top)), 4),
            "prob_after_random": round(float(sig(np.mean(rand_los))), 4),
            "removed": [int(i) + 1 for i in top]}


# ----------------------------------------------------------------- claims and questions

def _backend(kind: str):
    from generate_questions import OllamaBackend
    if kind == "local":
        return OllamaBackend()
    from commercial_backend import make_backend      # noqa: F401  (optional, key-gated)
    return make_backend()


def claims_and_questions(text: str, n_claims: int = 4, k_questions: int = 3,
                         backend: str = "local") -> dict:
    """Extract the submission's own claims, then write questions from each one.

    Claims carry the sentence numbers they came from, and the source text is looked up from the
    submission itself, so a quotation cannot be invented.
    """
    from generate_questions import sentences as split_sents, extract_claims, questions_for_claim, bloom_level
    t0 = time.time()
    be = _backend(backend)
    sents = split_sents(text)
    claims = extract_claims(sents, be, n_claims)
    out = []
    for c in claims:
        # extract_claims returns source_sentences as {"n", "text"} dicts, zero-indexed, and the
        # text is looked up from the submission rather than echoed by the model, which is what
        # makes an invented quotation impossible. Display numbers are 1-based for the reader.
        srcs = c.get("source_sentences", [])
        ns = [s["n"] + 1 for s in srcs]
        source = " ".join(s["text"] for s in srcs)
        qs = questions_for_claim(c["claim"], source, be, k=k_questions)
        out.append({"claim": c["claim"], "source_ns": ns, "source": source,
                    "questions": [{"q": q, "bloom": bloom_level(q)} for q in qs]})
    return {"claims": out, "n_sentences": len(sents), "seconds": round(time.time() - t0, 1)}


def status() -> dict:
    return {"ready": bool(_M.get("ready")),
            "hybrid": bool(_M.get("has_hybrid")),
            "load_seconds": _M.get("load_seconds")}
