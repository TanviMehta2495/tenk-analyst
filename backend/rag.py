"""Shared pieces for searching and storing 10-K chunks: models, Qdrant, filters."""
import functools
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langsmith import traceable
from qdrant_client import QdrantClient, models

load_dotenv()  # reads the .env file into environment variables

OPENROUTER = "https://openrouter.ai/api/v1"
COLLECTION = os.environ.get("QDRANT_COLLECTION", "tenk")


def get_embeddings() -> OpenAIEmbeddings:
    """Turns text into 1536 numbers. Goes through OpenRouter."""
    return OpenAIEmbeddings(
        model="openai/text-embedding-3-small",
        base_url=OPENROUTER,
        api_key=os.environ["OPENROUTER_API_KEY"],
        check_embedding_ctx_length=False,  # required for OpenRouter
    )


def get_client() -> QdrantClient:
    return QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"])


def get_llm() -> ChatOpenAI:
    """The chat model. temperature=0 makes the answers as repeatable as possible."""
    return ChatOpenAI(
        model=os.environ.get("CHAT_MODEL", "anthropic/claude-haiku-5.5"),
        base_url=OPENROUTER,
        api_key=os.environ["OPENROUTER_API_KEY"],
        temperature=0,
        max_tokens=1500,
    )


@functools.lru_cache(maxsize=1)
def get_store() -> QdrantVectorStore:
    """The Qdrant store. Built once and reused (lru_cache remembers the first result)."""
    return QdrantVectorStore(client=get_client(), collection_name=COLLECTION, embedding=get_embeddings())


@functools.lru_cache(maxsize=1)
def get_ranker():
    """The FlashRank model (about 3 MB, runs on the CPU). Loaded once, on first use."""
    from flashrank import Ranker
    return Ranker(model_name="ms-marco-TinyBERT-L-2-v2")


@traceable(name="flashrank_rerank")  # shows up as its own step in LangSmith
def rerank(question: str, docs: list, top_n: int = 5) -> list:
    """Re-score the chunks against the question and keep the best top_n, best first."""
    if not docs:
        return []
    from flashrank import RerankRequest
    passages = [{"id": i, "text": d.page_content} for i, d in enumerate(docs)]
    ranked = get_ranker().rerank(RerankRequest(query=question, passages=passages))
    return [docs[r["id"]] for r in ranked[:top_n]]


@traceable(run_type="retriever", name="qdrant_search")  # shows up as its own step in LangSmith
def search(question: str, k: int = 5, flt=None) -> list:
    """Return the k chunks whose meaning is closest to the question. flt limits companies/years/Items."""
    return get_store().similarity_search(question, k=k, filter=flt)


def make_filter(companies=None, years=None, items=None):
    """Build a Qdrant filter. An empty or missing list means 'do not filter on this'.

    LangChain stores metadata under 'metadata.', so the keys start with that.
    """
    must = []
    if companies:
        must.append(models.FieldCondition(key="metadata.company", match=models.MatchAny(any=list(companies))))
    if years:
        must.append(models.FieldCondition(key="metadata.year", match=models.MatchAny(any=[int(y) for y in years])))
    if items:
        must.append(models.FieldCondition(key="metadata.item", match=models.MatchAny(any=list(items))))
    return models.Filter(must=must) if must else None
