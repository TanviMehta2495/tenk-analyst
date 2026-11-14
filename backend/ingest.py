"""Stage 0: read the 10-K filings and store their chunks in Qdrant.

Run with:  uv run python -m backend.ingest
Safe to run twice: a filing that is already complete is skipped.
"""
import csv
import re
import uuid
from pathlib import Path

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import models

from backend.rag import COLLECTION, get_client, get_embeddings, make_filter

DATA = Path(__file__).resolve().parent.parent / "data"

# Each Item starts with a line like "## Item 1A. Risk Factors".
ITEM = re.compile(r"^## Item (\w+)\. (.+)$", re.M)

splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)


def read_items(path: Path) -> list[dict]:
    """Return one dict per Item: code, title, and text."""
    text = path.read_text(encoding="utf-8")
    heads = list(ITEM.finditer(text))
    items = []
    for i, h in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        items.append({"code": h.group(1), "title": h.group(2), "text": text[h.end():end].strip()})
    return items


def chunk_id(company: str, year: int, item: str, n: int) -> str:
    """A stable ID: the same chunk always gets the same ID, so a re-run overwrites instead of duplicating."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{company}/{year}/{item}/{n}"))


def chunk_filing(row: dict) -> tuple[list[Document], list[str]]:
    """Split one filing into chunks. Each Item is split on its own, so a chunk never crosses into the next Item."""
    ticker, year = row["ticker"], int(row["fiscal_year"])
    docs, ids = [], []
    for item in read_items(DATA / row["file"]):
        for n, piece in enumerate(splitter.split_text(item["text"])):
            docs.append(Document(
                page_content=piece,
                metadata={"company": ticker, "year": year, "item": item["code"],
                          "item_title": item["title"], "source": row["source_url"]},
            ))
            ids.append(chunk_id(ticker, year, item["code"], n))
    return docs, ids


def ensure_collection(client) -> None:
    """Create the collection once, with an index on each field we filter on."""
    if client.collection_exists(COLLECTION):
        return
    client.create_collection(COLLECTION, vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE))
    client.create_payload_index(COLLECTION, "metadata.company", models.PayloadSchemaType.KEYWORD)
    client.create_payload_index(COLLECTION, "metadata.item", models.PayloadSchemaType.KEYWORD)
    client.create_payload_index(COLLECTION, "metadata.year", models.PayloadSchemaType.INTEGER)


def main() -> None:
    client = get_client()
    ensure_collection(client)
    store = QdrantVectorStore(client=client, collection_name=COLLECTION, embedding=get_embeddings())

    with open(DATA / "filings.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        ticker, year = row["ticker"], int(row["fiscal_year"])
        docs, ids = chunk_filing(row)
        # Count what Qdrant already holds for this filing.
        stored = client.count(COLLECTION, count_filter=make_filter([ticker], [year]), exact=True).count
        if stored == len(docs):
            print(f"{ticker} {year}: skipped, {stored} chunks already stored")
            continue
        store.add_documents(docs, ids=ids, batch_size=200)
        print(f"{ticker} {year}: stored {len(docs)} chunks (had {stored})")

    total = client.count(COLLECTION, exact=True).count
    print(f"Done. The collection '{COLLECTION}' holds {total} chunks.")


if __name__ == "__main__":
    main()
