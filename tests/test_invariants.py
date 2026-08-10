"""Tests for the properties this dissertation actually claims.

The gate tests next door cover components. These cover the promises. Each one corresponds to a
sentence in the write-up that an examiner could reasonably ask me to back up, and each fails if
that sentence stops being true.

Everything here runs on committed artefacts, so anyone who clones the repository can run it. No
model weights, no corpus, no GPU, no network.

    python -m pytest tests/ -q
"""

import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "question_gen"))
sys.path.insert(0, str(REPO / "src" / "detection"))

import generate_questions as qg          # noqa: E402
from text_normalize import has_markup, normalize_text   # noqa: E402


# ---------------------------------------------------------------- provenance
# The claim: "the quoted text is looked up from the submission rather than echoed by the model, so
# an invented quotation is impossible by construction" (Sections 4.11, 7.4 and the guide's own
# front page). Impossible by construction is a strong word, so it gets a test.

ESSAY = [
    "Judicial review lets the courts check how public power is used.",
    "The doctrine developed case by case rather than by statute.",
    "Critics say it lets unelected judges second-guess elected ministers.",
    "That objection assumes review examines the merits, which it does not.",
    "The remedy is usually to send the decision back, not to replace it.",
]


class FakeBackend:
    """Stands in for a language model. Returns whatever the test tells it to."""

    def __init__(self, payload):
        self.payload = payload

    def chat_json(self, system, user):
        return self.payload


def test_quoted_text_always_comes_from_the_submission():
    backend = FakeBackend({"claims": [
        {"claim": "Review checks the use of power", "sentences": [0, 1]},
        {"claim": "The merits objection fails", "sentences": [3]},
    ]})
    claims = qg.extract_claims(ESSAY, backend, n=4)
    assert claims, "the fake backend returned usable claims, so some should survive"
    for c in claims:
        for s in c["source_sentences"]:
            assert s["text"] == ESSAY[s["n"]], "quoted text must be the submission's own sentence"
            assert s["text"] in ESSAY


def test_a_model_that_invents_a_quotation_cannot_get_one_through():
    """The model is asked for sentence numbers, never for text. Anything it invents is ignored."""
    backend = FakeBackend({"claims": [{
        "claim": "Invented",
        "sentences": [2],
        # A model trying to supply its own wording, which the pipeline never reads.
        "quote": "The court held that the minister acted unlawfully in every respect.",
        "text": "Something the student never wrote.",
    }]})
    claims = qg.extract_claims(ESSAY, backend, n=2)
    assert len(claims) == 1
    quoted = [s["text"] for s in claims[0]["source_sentences"]]
    assert quoted == [ESSAY[2]]
    joined = " ".join(quoted)
    assert "unlawfully" not in joined
    assert "never wrote" not in joined


def test_out_of_range_citations_are_dropped_rather_than_clamped():
    """Citing sentence 999 of a five-sentence essay must lose the claim, not quote sentence 4.
    Clamping would silently attach real text to a claim it did not come from."""
    backend = FakeBackend({"claims": [
        {"claim": "Cites a sentence that does not exist", "sentences": [999]},
        {"claim": "Cites a negative index", "sentences": [-1]},
        {"claim": "Cites one real sentence", "sentences": [1]},
    ]})
    claims = qg.extract_claims(ESSAY, backend, n=5)
    assert len(claims) == 1, "only the claim with a resolvable citation should survive"
    assert claims[0]["source_sentences"][0]["n"] == 1


def test_a_claim_with_no_usable_citation_is_dropped():
    backend = FakeBackend({"claims": [{"claim": "No provenance at all", "sentences": []}]})
    assert qg.extract_claims(ESSAY, backend, n=3) == []


# ---------------------------------------------------------------- the corpus cannot be gamed
# The claim: "if the AI essays were longer, the detector would learn length instead of style"
# (Section 4.6), and the feature model is therefore not allowed to see length at all.

def test_essay_length_is_never_a_feature():
    """n_words and n_sents are computed for reporting and dropped before training. If that ever
    stops being true, the detector gains exactly the shortcut the corpus design exists to close."""
    src = (REPO / "src" / "detection" / "hybrid_fusion.py").read_text(encoding="utf8")
    import ast
    tree = ast.parse(src)
    drop = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", None) == "DROP" for t in node.targets):
            drop = ast.literal_eval(node.value)
    assert drop is not None, "hybrid_fusion.py should define DROP"
    assert "n_words" in drop and "n_sents" in drop, f"length leaked into the feature set: {drop}"


def test_the_reported_feature_count_matches_the_code():
    """Appendix C says 23 stylometric features and a 24th input once perplexity is added. This
    caught the write-up claiming 25 in three places."""
    import ast
    src = (REPO / "src" / "detection" / "hybrid_fusion.py").read_text(encoding="utf8")
    drop = next(ast.literal_eval(n.value) for n in ast.walk(ast.parse(src))
                if isinstance(n, ast.Assign)
                and any(getattr(t, "id", None) == "DROP" for t in n.targets))
    sty = (REPO / "src" / "detection" / "stylometric.py").read_text(encoding="utf8")
    pos_tags = next(ast.literal_eval(n.value) for n in ast.walk(ast.parse(sty))
                    if isinstance(n, ast.Assign)
                    and any(getattr(t, "id", None) == "POS_TAGS" for t in n.targets))
    # Named in stylometric_features(), in the order the function assigns them.
    scalar = ["n_words", "n_sents", "mean_sent_len", "std_sent_len", "sent_len_cv", "burstiness",
              "ttr", "root_ttr", "hapax_ratio", "mean_word_len", "punct_ratio"]
    produced = scalar + [f"pos_{t}" for t in pos_tags]
    trained_on = [c for c in produced if c not in drop]
    assert len(trained_on) == 23, f"expected 23 trained features, got {len(trained_on)}"


# ---------------------------------------------------------------- splits are by writer
# The claim: "splits made at the level of the student, not the essay, so the same writer never
# appears on both sides" (Sections 4.5 and 3.2). The manifest is committed, so this is checkable
# by anyone.

def _manifest():
    path = REPO / "data" / "processed" / "bawe_human_sample_manifest.csv"
    with path.open(encoding="utf8", newline="") as fh:
        return list(csv.DictReader(fh))


def test_no_student_appears_in_two_splits():
    by_split = {}
    for row in _manifest():
        by_split.setdefault(row["split"], set()).add(row["student_id"])
    splits = sorted(by_split)
    assert len(splits) >= 2, "the manifest should record more than one split"
    for i, a in enumerate(splits):
        for b in splits[i + 1:]:
            shared = by_split[a] & by_split[b]
            assert not shared, f"{len(shared)} students appear in both {a} and {b}: {sorted(shared)[:5]}"


def test_the_sample_is_the_size_the_write_up_claims():
    rows = _manifest()
    assert len(rows) == 640, f"the write-up says 640 human essays, the manifest has {len(rows)}"
    assert len({r["id"] for r in rows}) == 640, "essay ids should be unique"


def test_every_cell_of_the_stratification_is_populated():
    """Four disciplinary groups crossed with first-language background, all eight non-empty. An
    empty cell would make the fairness breakdown in Section 6.2 impossible rather than merely
    noisy."""
    cells = {}
    for row in _manifest():
        cells[row["cell"]] = cells.get(row["cell"], 0) + 1
    assert len(cells) == 8, f"expected 8 cells, found {sorted(cells)}"
    assert min(cells.values()) > 0


# ---------------------------------------------------------------- the cleaning step
# The claim: a markup-only rule reached 92.5 percent before cleaning, so both classes are now
# normalised by identical code (Sections 3.4 and 4.9).

def test_bawe_export_tags_are_stripped_but_their_words_survive():
    raw = "<heading>1. Introduction</heading>For a new business, a plan is an outline."
    out = normalize_text(raw)
    assert not has_markup(out)
    assert "Introduction" in out and "outline" in out


def test_markdown_from_the_generator_is_stripped_too():
    raw = "**Business Description**\n\n* Basic: $0.99 per track\n1. **Track Sales**: revenue."
    out = normalize_text(raw)
    assert "**" not in out and "\n" not in out
    assert "Business Description" in out and "Track Sales" in out


def test_the_same_function_cleans_both_classes_to_the_same_shape():
    """The artefact existed because human text carried tags no AI text had. The test is that after
    normalisation neither class carries anything the other does not."""
    human = normalize_text("<heading>Method</heading>We surveyed forty students.")
    ai = normalize_text("## Method\n\nWe surveyed forty students.")
    assert human == ai == "Method We surveyed forty students."


def test_mathematical_comparisons_are_not_eaten_as_tags():
    """A naive tag regex deletes everything between a "<" and the next ">", which in academic prose
    means deleting the statistics. The text here spans both characters on purpose: a version with
    only "<" would pass under a broken regex too, because there would be nothing to close the
    match."""
    out = normalize_text("the effect held when p < 0.05 and the sample was n > 30 throughout")
    assert "p < 0.05" in out
    assert "n > 30" in out
    assert "the sample was" in out, "a greedy tag match would have swallowed the words between"
