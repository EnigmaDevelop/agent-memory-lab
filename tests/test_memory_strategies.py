from src.llm import LLMResponse, ScriptedLLM
from src.memory.base import SessionRecord
from src.memory.full import FullHistoryMemory
from src.memory.none import NoMemory
from src.memory.rag import RAGMemory
from src.memory.summary import RollingSummaryMemory


def make_record(session: int, content_len: int = 20, cited=None) -> SessionRecord:
    text = "detail " * content_len
    return SessionRecord(
        session=session,
        task_id=f"t{session}",
        task_prompt=f"Task body {text}",
        transcript=[{"role": "assistant", "content": "ack"}],
        final_answer="1",
        cited_source_session=cited,
    )


# --- NoMemory --------------------------------------------------------------


def test_no_memory_always_empty():
    mem = NoMemory()
    mem.on_session_end(make_record(1))
    ctx = mem.build_context({"prompt": "x"})
    assert ctx.text == ""
    assert ctx.included_sessions == []


# --- FullHistoryMemory -------------------------------------------------


def test_full_memory_no_history_is_empty():
    ctx = FullHistoryMemory().build_context({"prompt": "x"})
    assert ctx.text == ""
    assert ctx.included_sessions == []
    assert ctx.truncated is False


def test_full_memory_accumulates_in_chronological_order():
    mem = FullHistoryMemory(token_budget=100_000)
    for i in (1, 2, 3):
        mem.on_session_end(make_record(i))
    ctx = mem.build_context({"prompt": "x"})
    assert ctx.included_sessions == [1, 2, 3]
    assert ctx.truncated is False
    assert ctx.text.index("[Session 1]") < ctx.text.index("[Session 2]") < ctx.text.index("[Session 3]")


def test_full_memory_drops_oldest_first_when_over_budget():
    mem = FullHistoryMemory(token_budget=30)
    for i in (1, 2, 3, 4, 5):
        mem.on_session_end(make_record(i, content_len=40))
    ctx = mem.build_context({"prompt": "x"})
    assert ctx.truncated is True
    assert ctx.included_sessions == sorted(ctx.included_sessions)
    assert ctx.included_sessions[-1] == 5  # newest always survives
    assert 1 not in ctx.included_sessions  # oldest dropped first


# --- RollingSummaryMemory ------------------------------------------------


def test_summary_memory_generates_and_accumulates():
    llm = ScriptedLLM([LLMResponse(content="Summary A"), LLMResponse(content="Summary B")])
    mem = RollingSummaryMemory(llm)
    mem.on_session_end(make_record(3))
    mem.on_session_end(make_record(7))

    ctx = mem.build_context({"prompt": "x"})
    assert ctx.included_sessions == [3, 7]
    assert "[Session 3] Summary A" in ctx.text
    assert "[Session 7] Summary B" in ctx.text

    first_call_messages, first_call_tools = llm.calls[0]
    assert first_call_tools == []
    assert "Task body" in first_call_messages[0].content  # rendered session was actually sent


def test_summary_memory_no_history_is_empty():
    llm = ScriptedLLM([])
    ctx = RollingSummaryMemory(llm).build_context({"prompt": "x"})
    assert ctx.text == ""
    assert ctx.included_sessions == []


# --- RAGMemory -------------------------------------------------------------


def test_rag_memory_empty_index():
    ctx = RAGMemory().build_context({"prompt": "anything"})
    assert ctx.text == ""
    assert ctx.retrieved == []
    assert ctx.included_sessions == []


def test_rag_memory_retrieves_the_relevant_session():
    # BM25's classic idf formula is degenerate on a 2-3 document corpus (any
    # term in exactly one doc gets idf=0) — use enough filler sessions to
    # match the real experiment's corpus size and avoid that artifact.
    mem = RAGMemory(top_k=2)
    on_topic = SessionRecord(
        session=1,
        task_id="a",
        task_prompt="The team decided the high-value MRR threshold is $2000 for the loyalty program.",
        transcript=[],
        final_answer="6",
        cited_source_session=None,
    )
    filler_topics = [
        "How many support tickets currently have priority urgent across the company?",
        "What is the total invoice amount on record for customer Fathom Systems?",
        "How many employees are on the Support team right now?",
        "What is the combined attributed MRR for the Catalyst Summit campaign?",
        "How many customers signed up in the Manufacturing industry last year?",
    ]
    mem.on_session_end(on_topic)
    for i, prompt in enumerate(filler_topics, start=2):
        mem.on_session_end(
            SessionRecord(session=i, task_id=f"f{i}", task_prompt=prompt, transcript=[], final_answer="0", cited_source_session=None)
        )

    ctx = mem.build_context(
        {"prompt": "Using the high-value customer MRR threshold decided earlier, how many high-value customers are in Fintech?"}
    )
    assert ctx.retrieved
    assert ctx.retrieved[0]["session"] == 1
    assert 1 in ctx.included_sessions


def test_rag_memory_chunks_long_sessions_into_multiple_pieces():
    mem = RAGMemory()
    long_record = SessionRecord(
        session=1,
        task_id="a",
        task_prompt="word " * 200,
        transcript=[],
        final_answer="1",
        cited_source_session=None,
    )
    mem.on_session_end(long_record)
    assert len(mem.chunks) > 1
    assert all(c.session == 1 for c in mem.chunks)
