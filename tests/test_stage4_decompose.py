"""Stage 4 tests: decompose, parallel search, one combined answer."""
import json
from pathlib import Path

from backend.pipelines import MAX_SUB_QUESTIONS, MSG_COMPANY, answer, run_decompose

QUESTIONS = json.loads((Path(__file__).resolve().parent.parent / "test_data" / "questions.json").read_text())["questions"]


def test_question_3_sources_include_both_companies():
    item = next(x for x in QUESTIONS if x["id"] == 3)
    result = answer(item["question"], item["pipeline"], [], [], "test-stage4-q3")
    found = {(s["company"], s["year"]) for s in result["sources"]}
    assert ("TSLA", 2025) in found and ("F", 2025) in found
    assert result["answer"].strip() and result["tools_used"] == []
    assert len(result["sources"]) <= MAX_SUB_QUESTIONS * 3
    # no chunk is listed twice
    assert len({s["excerpt"] for s in result["sources"]}) == len(result["sources"])


def test_decompose_splits_by_company_and_keeps_years():
    subs = run_decompose("Compare Tesla's and Ford's tariff risks in 2025.")
    assert 2 <= len(subs) <= MAX_SUB_QUESTIONS
    assert {c for s in subs for c in s["companies"]} == {"TSLA", "F"}
    assert all(s["years"] == [2025] for s in subs)


def test_decompose_never_returns_more_than_4():
    subs = run_decompose("Compare risks of Tesla, GM, Ford, Rivian, Lucid, Aptiv and Carvana in 2024 and 2025.")
    assert 1 <= len(subs) <= MAX_SUB_QUESTIONS


def test_single_topic_question_stays_simple():
    subs = run_decompose("What does GM say about cybersecurity in 2024?")
    assert len(subs) <= 2
    assert all(s["companies"] in ([], ["GM"]) for s in subs)


def test_out_of_scope_company_stops_before_searching():
    result = answer("Compare Tesla and Toyota tariff risks in 2025.", "decompose", [], [], "test-stage4-toyota")
    assert result["answer"] == MSG_COMPANY and result["sources"] == []


def test_sidebar_conflict_stops_before_searching():
    result = answer("Compare Tesla's and Ford's tariff risks in 2025.", "decompose", ["TSLA"], [], "test-stage4-sb")
    assert "not selected in the sidebar" in result["answer"] and "Ford" in result["answer"]
    assert result["sources"] == []


def test_sidebar_limits_every_sub_question():
    result = answer("Compare tariff risks of Tesla and Ford.", "decompose", ["TSLA", "F"], [2025], "test-stage4-limit")
    assert result["sources"]
    assert all(s["company"] in ("TSLA", "F") and s["year"] == 2025 for s in result["sources"])
