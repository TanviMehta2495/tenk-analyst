"""Stage 5 tests: SEC data, the two tools, and the agent."""
import json
import uuid
from pathlib import Path

from langgraph.errors import GraphRecursionError

from backend import agent as agent_module
from backend.agent import build_tools
from backend.pipelines import MSG_COMPANY, MSG_NOT_REPORTED, MSG_TOO_MANY_STEPS, answer
from backend.sec_data import annual_value, format_money

QUESTIONS = json.loads((Path(__file__).resolve().parent.parent / "test_data" / "questions.json").read_text())["questions"]


def q(n):
    return next(x for x in QUESTIONS if x["id"] == n)


def tools_for(sidebar_companies=(), sidebar_years=()):
    used, sources = [], []
    search_10k, sec_tool = build_tools(list(sidebar_companies), list(sidebar_years), used, sources)
    return sec_tool, search_10k, used, sources


# ---- no LLM: SEC data and the tools

def test_format_money():
    assert format_money(9_400_000_000, 2025) == "$9.4 billion (FY2025)"
    assert format_money(-3_700_000_000, 2025) == "-$3.7 billion (FY2025)"


def test_sec_data_checked_values():
    assert round(annual_value("F", "revenue", 2024)[0] / 1e9, 1) == 185.0
    assert round(annual_value("F", "rd_expense", 2025)[0] / 1e9, 1) == 9.4
    assert annual_value("GM", "rd_expense", 2025) == (None, None)


def test_tool_returns_ford_revenue_2024_and_a_source():
    sec_tool, _, used, sources = tools_for()
    text = sec_tool.invoke({"ticker": "F", "metric": "revenue", "years": [2024]})
    assert "$185.0 billion (FY2024)" in text
    assert used == ["sec_financial_data"]
    assert sources[0]["type"] == "sec" and sources[0]["company"] == "F" and sources[0]["year"] == 2024
    assert sources[0]["metric"] == "revenue" and sources[0]["tag"] and sources[0]["value"] > 1e11


def test_tool_does_not_return_gm_rd_2025():
    sec_tool, _, _, sources = tools_for()
    text = sec_tool.invoke({"ticker": "GM", "metric": "rd_expense", "years": [2025]})
    assert "not in the SEC financial data" in text and "$" not in text
    assert sources == []  # no number, so no source


def test_tool_rejects_unsupported_input_with_a_list_of_what_is_supported():
    sec_tool, _, _, sources = tools_for()
    for args in ({"ticker": "TM", "metric": "revenue", "years": [2025]},
                 {"ticker": "F", "metric": "ebitda", "years": [2025]},
                 {"ticker": "F", "metric": "revenue", "years": [2023]}):
        assert "Not supported" in sec_tool.invoke(args)
    assert sources == []


def test_tool_respects_the_sidebar():
    sec_tool, search_10k, _, sources = tools_for(["TSLA"], [2025])
    assert "sidebar does not include Ford" in sec_tool.invoke({"ticker": "F", "metric": "revenue", "years": [2025]})
    assert "sidebar does not include FY2024" in sec_tool.invoke({"ticker": "TSLA", "metric": "revenue", "years": [2024]})
    assert "sidebar does not include" in search_10k.invoke({"query": "risks", "companies": ["F"]})
    assert sources == []


def test_search_tool_cannot_widen_the_sidebar():
    _, search_10k, used, sources = tools_for(["F"], [2025])
    search_10k.invoke({"query": "cybersecurity risk"})  # no companies given: the sidebar is the limit
    assert used == ["search_10k"] and len(sources) == 5
    assert all(s["company"] == "F" and s["year"] == 2025 for s in sources)


def test_step_limit_gives_the_too_many_steps_message(monkeypatch):
    class Stuck:
        def invoke(self, *args, **kwargs):
            raise GraphRecursionError("limit")

    monkeypatch.setattr(agent_module, "create_agent", lambda *a, **k: Stuck())
    result = answer("What was Ford's R&D expense in 2025?", "agent", [], [], str(uuid.uuid4()))
    assert result["answer"] == MSG_TOO_MANY_STEPS


# ---- with the LLM

def test_question_4_ford_rd():
    result = answer(q(4)["question"], "agent", [], [], str(uuid.uuid4()))
    assert "9.4" in result["answer"]
    assert "sec_financial_data" in result["tools_used"]
    assert any(s["type"] == "sec" and s["company"] == "F" and s["year"] == 2025 for s in result["sources"])


def test_question_5_gm_rd_comes_from_the_10k_text():
    result = answer(q(5)["question"], "agent", [], [], str(uuid.uuid4()))
    assert "8.5" in result["answer"]
    assert any(s["type"] == "10-K" and s["company"] == "GM" and s["year"] == 2025 for s in result["sources"])


def test_question_6_carvana_rd_is_not_reported():
    result = answer(q(6)["question"], "agent", [], [], str(uuid.uuid4()))
    assert MSG_NOT_REPORTED in result["answer"]
    assert "sec_financial_data" in result["tools_used"]


def test_question_7_toyota_is_out_of_scope():
    result = answer(q(7)["question"], "agent", [], [], str(uuid.uuid4()))
    assert result["answer"] == MSG_COMPANY
    assert result["sources"] == [] and result["tools_used"] == []


def test_question_10_sidebar_conflict():
    result = answer("What was GM's R&D expense in 2025?", "agent", ["TSLA", "F"], [], str(uuid.uuid4()))
    assert "not selected in the sidebar" in result["answer"] and "General Motors" in result["answer"]
    assert result["sources"] == [] and result["tools_used"] == []
