"""Exact numbers from SEC's XBRL company facts (the structured data behind every 10-K)."""
import csv
import functools
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

DATA = Path(__file__).resolve().parent.parent / "data"

# metric name -> XBRL tags to try, in this order
METRICS = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "rd_expense": ["ResearchAndDevelopmentExpense", "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost"],
    "operating_income": ["OperatingIncomeLoss"],
}


@functools.lru_cache(maxsize=1)
def ciks() -> dict[str, str]:
    """ticker -> CIK (SEC's company number), read from data/filings.csv."""
    with open(DATA / "filings.csv", newline="", encoding="utf-8") as f:
        return {row["ticker"]: row["cik"] for row in csv.DictReader(f)}


@functools.lru_cache(maxsize=32)  # one big download per company, then it stays in memory
def company_facts(cik: str) -> dict:
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik.zfill(10)}.json"
    # SEC requires a User-Agent with a name and email. It comes from the environment, never from the code.
    response = httpx.get(url, headers={"User-Agent": os.environ["SEC_USER_AGENT"]}, timeout=60)
    response.raise_for_status()
    return response.json()


def annual_value(ticker: str, metric: str, year: int):
    """Return (value, tag) for a full-year amount, or (None, None) if the company does not report it."""
    facts = company_facts(ciks()[ticker])["facts"].get("us-gaap", {})
    for tag in METRICS[metric]:
        for values in facts.get(tag, {}).get("units", {}).values():
            for v in values:
                # 'frame' CY2025 means the calendar year 2025. Each 10-K also repeats older years; the frame skips those.
                if v.get("frame") == f"CY{year}":
                    return v["val"], tag
    return None, None


def format_money(value: float, year: int) -> str:
    """Billions with one decimal and the fiscal year. A loss has the minus sign before the dollar sign."""
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value) / 1e9:.1f} billion (FY{year})"
