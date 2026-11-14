"""Stage 2 tests: the filters pipeline and the sidebar rules."""
import json
from pathlib import Path

from backend.pipelines import answer, resolve_filters, sidebar_conflict

QUESTIONS = json.loads((Path(__file__).resolve().parent.parent / "test_data" / "questions.json").read_text())["questions"]


def q(n):
    return next(x for x in QUESTIONS if x["id"] == n)


# ---- no LLM: the rules on their own

def test_sidebar_replaces_only_its_own_list():
    state = {"companies": ["GM"], "years": [2024], "sidebar_companies": ["GM", "F"], "sidebar_years": []}
    assert resolve_filters(state) == (["GM", "F"], [2024])  # companies from the sidebar, years from the question


def test_nothing_set_means_search_everything():
    assert resolve_filters({"companies": [], "years": []}) == ([], [])


def test_sidebar_conflict_names_what_is_missing():
    msg = sidebar_conflict(["GM"], [2025], ["TSLA", "F"], [])
    assert msg == ("This question asks about something that is not selected in the sidebar: General Motors. "
                   "Select it in the sidebar, or change the question.")
    both = sidebar_conflict(["GM", "F"], [2024, 2025], ["F"], [2025])
    assert "General Motors" in both and "FY2024" in both and "Ford" not in both
    assert sidebar_conflict(["GM"], [], [], []) == ""  # empty sidebar = not set = no conflict


# ---- with the LLM

def test_question_2_all_sources_are_gm_2024():
    item = q(2)
    result = answer(item["question"], item["pipeline"], [], [], "test-stage2-q2")
    assert result["sources"]
    assert all(s["company"] == "GM" and s["year"] == 2024 for s in result["sources"])


def test_question_9_sidebar_conflict():
    item = q(9)
    result = answer("What was GM's R&D expense in 2025?", "filters", ["TSLA", "F"], [], "test-stage2-q9")
    assert "not selected in the sidebar" in result["answer"] and "General Motors" in result["answer"]
    assert result["sources"] == [] and result["tools_used"] == []


def test_sidebar_limits_the_search():
    # The question names no company. The sidebar says Ford only, so every source must be Ford.
    result = answer("What does the company say about cybersecurity?", "filters", ["F"], [2025], "test-stage2-sb")
    assert result["sources"]
    assert all(s["company"] == "F" and s["year"] == 2025 for s in result["sources"])


def test_simple_ignores_the_sidebar():
    result = answer("What was GM's R&D expense in 2025?", "simple", ["TSLA"], [], "test-stage2-simple")
    assert "not selected in the sidebar" not in result["answer"]
    assert len(result["sources"]) == 5
