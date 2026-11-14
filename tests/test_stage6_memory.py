"""Stage 6 tests: the agent remembers a conversation in Neon."""
import uuid

import pytest

from backend import agent as agent_module
from backend.pipelines import answer


@pytest.fixture
def chat_id():
    """A NEW chat ID on every run, so an old conversation in Neon can never change the result.
    The thread is deleted from Neon afterwards."""
    cid = str(uuid.uuid4())
    yield cid
    agent_module.get_checkpointer().delete_thread(cid)


def saved_messages(cid: str) -> int:
    saved = agent_module.get_checkpointer().get_tuple({"configurable": {"thread_id": cid}})
    return len(saved.checkpoint["channel_values"]["messages"]) if saved else 0


def test_question_8_follow_up_uses_the_previous_question(chat_id):
    first = answer("What was Ford's revenue in 2024?", "agent", [], [], chat_id)
    second = answer("And in 2025?", "agent", [], [], chat_id)
    assert "185.0" in first["answer"]
    assert "187.3" in second["answer"]
    # tools_used and sources describe only the latest turn
    assert second["tools_used"] == ["sec_financial_data"]
    assert [(s["company"], s["year"]) for s in second["sources"]] == [("F", 2025)]


def test_conversation_is_saved_in_neon_and_survives_a_restart(chat_id):
    answer("What was Ford's revenue in 2024?", "agent", [], [], chat_id)
    before = saved_messages(chat_id)
    assert before >= 2  # at least the question and the answer
    agent_module.close_memory()  # like stopping the back end: the pool is closed and made again
    reply = answer("Which company did I ask about?", "agent", [], [], chat_id)
    assert "Ford" in reply["answer"]
    assert saved_messages(chat_id) > before


def test_a_new_chat_does_not_see_the_old_one(chat_id):
    answer("What was Ford's revenue in 2024?", "agent", [], [], chat_id)
    fresh = str(uuid.uuid4())
    try:
        reply = answer("And in 2025?", "agent", [], [], fresh)
        assert "187.3" not in reply["answer"] and "185.0" not in reply["answer"]
    finally:
        agent_module.get_checkpointer().delete_thread(fresh)


def test_out_of_scope_follow_up_is_still_refused(chat_id):
    answer("What was Ford's revenue in 2024?", "agent", [], [], chat_id)
    reply = answer("And Toyota?", "agent", [], [], chat_id)
    assert reply["sources"] == [] and reply["tools_used"] == []
    assert reply["answer"].startswith("This app covers only")


def test_other_pipelines_have_no_memory(chat_id):
    answer("What was Ford's revenue in 2024?", "filters", [], [], chat_id)
    assert saved_messages(chat_id) == 0  # nothing was written to Neon for this chat
