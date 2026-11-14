"""The Streamlit page. It talks to the back end only over HTTP.

Start it with:  uv run python -m streamlit run frontend/app.py
(The back end must be running too: uv run python -m uvicorn backend.api:app --port 8000)
"""
import hmac
import os
import uuid

import httpx
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
TIMEOUT = 90  # seconds. A sleeping free server on Render needs about 30 seconds to wake up.

COMPANIES = {
    "TSLA": "Tesla", "GM": "General Motors", "F": "Ford", "RIVN": "Rivian", "LCID": "Lucid",
    "APTV": "Aptiv", "BWA": "BorgWarner", "ALV": "Autoliv", "HOG": "Harley-Davidson", "CVNA": "Carvana",
}
YEARS = [2024, 2025]
PIPELINES = {
    "agent": "agent: tools, SEC data, and memory",
    "simple": "simple: search all chunks",
    "filters": "filters: search only the companies and years asked",
    "rerank": "rerank: 20 chunks, keep the best 5",
    "decompose": "decompose: split the question, search each part",
}
EXAMPLES = [
    "What risks does Tesla report about battery supply?",
    "Compare GM and Ford R&D in 2025",
    "How did Carvana's revenue change from 2024 to 2025?",
]
MSG_WAKING = "The server is waking up. This takes about 30 seconds."
MSG_UNAVAILABLE = "The answer service is not available right now. Try again in a minute."

st.set_page_config(page_title="10-K Analyst", page_icon="📈")


# ---------------------------------------------------------------- the access code screen

def access_screen() -> bool:
    """Show the code screen until the right code is entered. Returns True when the user may continue."""
    if st.session_state.get("authed"):  # st.session_state remembers things between reruns of this script
        return True
    expected = os.environ.get("ACCESS_CODE", "")
    st.title("10-K Analyst")
    if not expected:
        st.error("ACCESS_CODE is not set for this page. Add it to your settings and restart.")
        return False
    with st.form("access"):  # a form lets the Enter key submit, as well as the button
        code = st.text_input("Access code", type="password")
        submitted = st.form_submit_button("Enter")
    if submitted:
        if hmac.compare_digest(code.encode(), expected.encode()):
            st.session_state["authed"] = True
            st.rerun()
        st.error("Wrong access code.")
    return False


# ---------------------------------------------------------------- talking to the back end

def call_api(question: str, pipeline: str, companies: list[str], years: list[int]) -> dict:
    """Ask the back end. Returns its JSON. If the server is asleep, wait and try once more."""
    payload = {"question": question, "chat_id": st.session_state.chat_id, "pipeline": pipeline,
               "companies": companies, "years": years}
    headers = {"X-Access-Code": os.environ["ACCESS_CODE"]}
    for attempt in (1, 2):
        try:
            response = httpx.post(f"{API_URL}/ask", json=payload, headers=headers, timeout=TIMEOUT)
            return response.json()  # a good answer, or {"error": "..."} that we show as it is
        except (httpx.TimeoutException, httpx.TransportError):
            if attempt == 1:
                st.info(MSG_WAKING)
    return {"error": MSG_UNAVAILABLE}


# ---------------------------------------------------------------- showing an answer

def money(value: float, year: int) -> str:
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value) / 1e9:.1f} billion (FY{year})"


def show_sources(sources: list[dict]) -> None:
    with st.expander(f"Sources ({len(sources)})"):
        for s in sources:
            name = COMPANIES.get(s["company"], s["company"])
            if s["type"] == "sec":
                st.markdown(f"**{name}, FY{s['year']}, SEC data:** {s['metric']} = {money(s['value'], s['year'])}  \n"
                            f"SEC tag: `{s['tag']}`")
            else:
                st.markdown(f"**{name}, FY{s['year']} 10-K, Item {s['item']}** ({s['item_title']})  \n"
                            f"{s['excerpt']}…  \n[Open the filing]({s['url']})")


def show_message(message: dict, show_source_list: bool) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["text"])
        if message["role"] == "assistant" and not message.get("is_error"):
            tools = message.get("tools_used", [])
            st.caption("Tools used: " + (" · ".join(f"`{t}`" for t in tools) if tools else "none"))
            if show_source_list and message.get("sources"):
                show_sources(message["sources"])


# ---------------------------------------------------------------- the page

def new_chat() -> None:
    st.session_state.chat_id = str(uuid.uuid4())  # a new ID = a new conversation (and a new memory)
    st.session_state.messages = []


def main() -> None:
    if not access_screen():
        return
    if "chat_id" not in st.session_state:
        new_chat()

    with st.sidebar:
        st.header("Filters")
        picked_companies = st.multiselect("Companies", list(COMPANIES), default=list(COMPANIES),
                                          format_func=COMPANIES.get)
        picked_years = st.multiselect("Fiscal years", YEARS, default=YEARS)
        show_source_list = st.toggle("Show sources", value=True)
        st.button("New chat", on_click=new_chat)
        with st.expander("Learning mode", expanded=False):
            pipeline = st.radio("Pipeline", list(PIPELINES), format_func=PIPELINES.get, index=0)
            st.caption("Only the agent uses SEC data and remembers the chat.")

    # Everything selected means "no filter", so we send an empty list. An empty selection also means no filter.
    companies = [] if len(picked_companies) in (0, len(COMPANIES)) else picked_companies
    years = [] if len(picked_years) in (0, len(YEARS)) else picked_years

    st.title("10-K Analyst")
    st.caption("Questions about the 10-K filings of ten US auto companies, fiscal years 2024 and 2025.")

    for message in st.session_state.messages:
        show_message(message, show_source_list)

    # A clicked example is saved in 'pending', and the page reruns. Then the buttons are not drawn at all.
    question = st.session_state.pop("pending", None)
    if not st.session_state.messages and not question:
        for i, text in enumerate(EXAMPLES):
            if st.button(text, key=f"example{i}"):
                st.session_state.pending = text
                st.rerun()
    typed = st.chat_input("Ask about a 10-K filing")  # draw it in EVERY run, or the box disappears
    question = question or typed
    if not question:
        return

    user_message = {"role": "user", "text": question}
    show_message(user_message, show_source_list)
    with st.spinner("Searching the filings…"):
        result = call_api(question, pipeline, companies, years)

    if "error" in result:
        reply = {"role": "assistant", "text": result["error"], "is_error": True}
    else:
        reply = {"role": "assistant", "text": result["answer"], "tools_used": result["tools_used"],
                 "sources": result["sources"]}
    show_message(reply, show_source_list)
    st.session_state.messages += [user_message, reply]


main()
