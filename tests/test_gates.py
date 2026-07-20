"""Unit tests for the pipeline's gates and hand-rolled statistics.

These are the pieces the evaluation chain leans on hardest: the well-formedness gate that keeps
degenerate questions away from scores and lecturers, the content gate behind the v4 experiment,
and the Krippendorff alpha implementation used for judge agreement. Each earned a test the hard
way, by catching or measuring a real failure during the project.

    python -m pytest tests/ -q
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "question_gen"))
sys.path.insert(0, str(REPO / "src" / "evaluation"))


# ---- well-formedness gate (src/question_gen/wellformed.py) ----

def test_degenerate_mcq_stem_is_caught():
    from wellformed import is_degenerate
    assert is_degenerate("Which of the following is correct?")


def test_real_verification_question_passes():
    from wellformed import well_formed
    q = ("How did you decide which evidence to include for this claim, "
         "and why that evidence over the alternatives?")
    assert well_formed(q)


def test_non_questions_fail():
    from wellformed import well_formed
    assert not well_formed("This is a statement, not a question.")
    assert not well_formed("Why?")  # too short to be a usable interview question


# ---- content gate (src/question_gen/build_v4_dataset.py) ----

CLAIM = "The communisation of Eastern Europe was not a foregone conclusion in May 1945."
SOURCE = "Local actors in Poland and Hungary resisted Soviet influence after the war."


def test_content_word_leak_is_rejected():
    from build_v4_dataset import content_free
    q = "What led you to argue that communisation was not inevitable?"
    assert not content_free(q, CLAIM, SOURCE)


def test_entity_leak_from_source_is_rejected():
    from build_v4_dataset import content_free
    q = "How do events in Poland and Hungary support your position?"
    assert not content_free(q, CLAIM, SOURCE)


def test_anchored_content_free_question_passes():
    from build_v4_dataset import content_free
    q = ("Walk me through the reasoning that led you to this claim, "
         "and the evidence you trusted most.")
    assert content_free(q, CLAIM, SOURCE)


def test_anchor_words_are_allowed():
    from build_v4_dataset import content_free
    q = "If this claim turned out to be wrong, which part of your essay would suffer most?"
    assert content_free(q, CLAIM, SOURCE)


# ---- Krippendorff alpha, interval metric (src/evaluation/llm_judge.py) ----

def test_alpha_perfect_agreement_is_one():
    from llm_judge import krippendorff_alpha_interval
    matrix = [[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]
    assert abs(krippendorff_alpha_interval(matrix) - 1.0) < 1e-9


def test_alpha_inverted_ratings_are_negative():
    from llm_judge import krippendorff_alpha_interval
    matrix = [[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]]
    assert krippendorff_alpha_interval(matrix) < 0


def test_alpha_handles_missing_values():
    from llm_judge import krippendorff_alpha_interval
    matrix = [[1.0, 2.0, None, 4.0], [1.0, 2.0, 3.0, 4.0]]
    a = krippendorff_alpha_interval(matrix)
    assert abs(a - 1.0) < 1e-9  # items with fewer than two ratings are excluded


def test_alpha_constant_offset_hurts_interval_agreement():
    from llm_judge import krippendorff_alpha_interval
    same = [[3.0, 3.5, 4.0, 4.5], [3.0, 3.5, 4.0, 4.5]]
    offset = [[3.0, 3.5, 4.0, 4.5], [4.0, 4.5, 5.0, 5.5]]
    assert krippendorff_alpha_interval(offset) < krippendorff_alpha_interval(same)
