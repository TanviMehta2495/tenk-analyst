"""Stage 0 tests: the ingest command fills Qdrant, and a second run changes nothing."""
import re
import subprocess
import sys
from pathlib import Path

from backend.ingest import DATA, chunk_filing, read_items
from backend.rag import COLLECTION, get_client, make_filter

ROOT = Path(__file__).resolve().parent.parent
EXPECTED = 10229


def count() -> int:
    return get_client().count(COLLECTION, exact=True).count


def run_ingest() -> str:
    result = subprocess.run([sys.executable, "-m", "backend.ingest"], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-500:]
    return result.stdout


def test_collection_has_all_chunks():
    # The first run embeds everything (about a minute). Later runs only count.
    if count() != EXPECTED:
        run_ingest()
    assert count() == EXPECTED


def test_second_run_changes_nothing():
    before = count()
    out = run_ingest()
    assert count() == before == EXPECTED
    # Every filing must be skipped: no line says "stored N chunks (had ...)".
    assert out.count("skipped") == 20
    assert "(had" not in out


def test_chunk_metadata_is_complete():
    client = get_client()
    points, _ = client.scroll(COLLECTION, scroll_filter=make_filter(["F"], [2025], ["7"]), limit=5, with_payload=True)
    assert points
    for p in points:
        meta = p.payload["metadata"]
        assert meta["company"] == "F" and meta["year"] == 2025 and meta["item"] == "7"
        assert meta["item_title"] and meta["source"].startswith("https://www.sec.gov/")


def test_chunks_never_cross_items():
    # No chunk may contain another Item heading, and chunks are at most about 1,500 characters.
    heading = re.compile(r"^## Item \w+\. ", re.M)
    row = {"ticker": "F", "fiscal_year": "2025", "source_url": "u", "file": "10k/F_2025.md"}
    docs, _ = chunk_filing(row)
    assert len(read_items(DATA / row["file"])) > 5  # the filing really splits into several Items
    for d in docs:
        assert not heading.search(d.page_content)
        assert len(d.page_content) <= 1500
