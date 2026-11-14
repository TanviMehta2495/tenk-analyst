"""The FastAPI back end.

Start it with:  uv run python -m uvicorn backend.api:app --port 8000
"""
import hmac
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from backend.agent import close_memory
from backend.pipelines import PIPELINES, answer

logger = logging.getLogger("tenk.api")

REQUIRED = ["OPENROUTER_API_KEY", "QDRANT_URL", "QDRANT_API_KEY", "NEON_DATABASE_URL", "SEC_USER_AGENT", "ACCESS_CODE"]
OPTIONAL_BUT_CHECKED = ["LANGSMITH_API_KEY"]  # tracing is optional, but a value with a line break would break it
SECRETS = REQUIRED + OPTIONAL_BUT_CHECKED + ["LANGSMITH_TRACING"]

ERR_CODE = "Wrong or missing access code."
ERR_EMPTY = "The question is empty."
ERR_LONG = "The question is too long. Use at most 1,000 characters."
ERR_PIPELINE = "Unknown pipeline. Use simple, filters, rerank, decompose, or agent."
ERR_UNAVAILABLE = "The answer service is not available right now. Try again in a minute."
ERR_REQUEST = "The request is not valid."


def check_environment() -> None:
    """Stop at start-up if a setting is missing or has a line break. Print only the NAME, never the value.

    A key pasted with an extra line would fail later, and the library's error message could print the key.
    """
    for name in REQUIRED:
        if not os.environ.get(name):
            raise SystemExit(f"Missing environment variable: {name}")
    for name in REQUIRED + OPTIONAL_BUT_CHECKED:
        value = os.environ.get(name, "")
        if "\n" in value or "\r" in value:
            raise SystemExit(f"The environment variable {name} contains a line break. Fix it and start again.")


def hide_secrets(text: str) -> str:
    """Replace any secret value inside a log message, so a key or a password never reaches the logs."""
    for name in ["OPENROUTER_API_KEY", "QDRANT_API_KEY", "NEON_DATABASE_URL", "LANGSMITH_API_KEY", "ACCESS_CODE"]:
        value = os.environ.get(name)
        if value:
            text = text.replace(value, "[hidden]")
    return text


@asynccontextmanager
async def lifespan(app: FastAPI):
    check_environment()
    yield
    close_memory()  # close the database connections when the back end stops


app = FastAPI(title="10-K Analyst", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = ""
    chat_id: str = ""
    pipeline: str = "agent"
    companies: list[str] = []
    years: list[int] = []


def error(status: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": message})


@app.exception_handler(RequestValidationError)
async def bad_request(request: Request, exc: RequestValidationError):
    return error(400, ERR_REQUEST)  # our own short message, not FastAPI's long default


@app.get("/health")
def health():
    return {"status": "ok"}  # calls no paid service


# 'def' (not 'async def') makes FastAPI run this in a worker thread, so one slow answer does not block others.
@app.post("/ask")
def ask(body: AskRequest, x_access_code: str | None = Header(default=None)):
    expected = os.environ.get("ACCESS_CODE", "")
    # compare_digest compares in constant time. An empty expected code never matches.
    if not expected or not x_access_code or not hmac.compare_digest(x_access_code.encode(), expected.encode()):
        return error(401, ERR_CODE)
    question = body.question.strip()
    if not question:
        return error(400, ERR_EMPTY)
    if len(question) > 1000:
        return error(400, ERR_LONG)
    if body.pipeline not in PIPELINES:
        return error(400, ERR_PIPELINE)
    try:
        return answer(question, body.pipeline, body.companies, body.years, body.chat_id)
    except Exception as e:  # OpenRouter, Qdrant, Neon, or SEC failed
        # Log the error type and a cleaned message. Never send details to the client.
        logger.error("answer failed: %s: %s", type(e).__name__, hide_secrets(str(e))[:500])
        return error(503, ERR_UNAVAILABLE)
