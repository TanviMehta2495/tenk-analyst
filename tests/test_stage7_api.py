"""Stage 7 tests: the API. None of these call a paid service (the answer function is replaced where needed)."""
import logging
import os

import pytest
from fastapi.testclient import TestClient

from backend import api
from backend.api import ERR_CODE, ERR_EMPTY, ERR_LONG, ERR_PIPELINE, ERR_UNAVAILABLE, REQUIRED, check_environment

client = TestClient(api.app)  # no 'with': the start-up check is tested on its own below


def headers():
    return {"X-Access-Code": os.environ["ACCESS_CODE"]}


def body(**changes):
    return {"question": "What was Ford's R&D expense in 2025?", "chat_id": "c1", "pipeline": "agent",
            "companies": [], "years": [], **changes}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200 and response.json() == {"status": "ok"}


def test_missing_access_code():
    response = client.post("/ask", json=body())
    assert response.status_code == 401 and response.json() == {"error": ERR_CODE}


def test_wrong_access_code():
    response = client.post("/ask", json=body(), headers={"X-Access-Code": "not-the-code"})
    assert response.status_code == 401 and response.json() == {"error": ERR_CODE}


@pytest.mark.parametrize("question", ["", "   "])
def test_empty_question(question):
    response = client.post("/ask", json=body(question=question), headers=headers())
    assert response.status_code == 400 and response.json() == {"error": ERR_EMPTY}


def test_too_long_question():
    response = client.post("/ask", json=body(question="x" * 1001), headers=headers())
    assert response.status_code == 400 and response.json() == {"error": ERR_LONG}


def test_unknown_pipeline():
    response = client.post("/ask", json=body(pipeline="magic"), headers=headers())
    assert response.status_code == 400 and response.json() == {"error": ERR_PIPELINE}


def test_access_code_is_checked_first():
    response = client.post("/ask", json=body(question="", pipeline="magic"))
    assert response.status_code == 401


def test_success_passes_everything_to_answer(monkeypatch):
    seen = {}

    def fake(question, pipeline, companies, years, chat_id):
        seen.update(question=question, pipeline=pipeline, companies=companies, years=years, chat_id=chat_id)
        return {"answer": "ok", "pipeline": pipeline, "chat_id": chat_id, "tools_used": [], "sources": []}

    monkeypatch.setattr(api, "answer", fake)
    response = client.post("/ask", json=body(pipeline="filters", companies=["F"], years=[2025]), headers=headers())
    assert response.status_code == 200
    assert response.json() == {"answer": "ok", "pipeline": "filters", "chat_id": "c1", "tools_used": [], "sources": []}
    assert seen == {"question": "What was Ford's R&D expense in 2025?", "pipeline": "filters",
                    "companies": ["F"], "years": [2025], "chat_id": "c1"}


def test_service_failure_gives_503_without_details_and_without_secrets(monkeypatch, caplog):
    secret = os.environ["OPENROUTER_API_KEY"]

    def broken(*args):
        raise RuntimeError(f"Connection failed, key {secret}")

    monkeypatch.setattr(api, "answer", broken)
    with caplog.at_level(logging.ERROR, logger="tenk.api"):
        response = client.post("/ask", json=body(), headers=headers())
    assert response.status_code == 503 and response.json() == {"error": ERR_UNAVAILABLE}
    assert secret not in response.text
    assert "RuntimeError" in caplog.text and secret not in caplog.text  # logged, but the key is hidden


# ---- the start-up check

def test_startup_stops_on_a_missing_variable_and_names_it(monkeypatch):
    for name in REQUIRED:
        monkeypatch.setenv(name, "value")
    monkeypatch.delenv("SEC_USER_AGENT")
    with pytest.raises(SystemExit) as stop:
        check_environment()
    assert str(stop.value) == "Missing environment variable: SEC_USER_AGENT"


@pytest.mark.parametrize("name", ["QDRANT_API_KEY", "LANGSMITH_API_KEY"])
def test_startup_stops_on_a_line_break_and_never_prints_the_value(monkeypatch, name):
    for n in REQUIRED:
        monkeypatch.setenv(n, "value")
    monkeypatch.setenv(name, "topsecret-123\nextra")
    with pytest.raises(SystemExit) as stop:
        check_environment()
    assert name in str(stop.value) and "topsecret" not in str(stop.value)


def test_startup_passes_when_everything_is_set():
    check_environment()  # the real settings from .env: no error means every variable is set
