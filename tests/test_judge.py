from src.judge import is_correct, judge_session, normalize_numeric, normalize_string
from src.memory.base import SessionRecord

INT_TASK = {"task_type": "dependent", "answer_type": "int", "answer": 6}
STRING_TASK = {"task_type": "dependent", "answer_type": "string", "answer": "yes"}
TRAP_TASK = {
    "task_type": "trap",
    "answer_type": "int",
    "answer": 22,
    "stale_answer": 28,
    "expected_source_session": 25,
    "stale_source_session": 17,
}


def record(final_answer, cited=None, session=30) -> SessionRecord:
    return SessionRecord(
        session=session, task_id="x", task_prompt="p", transcript=[], final_answer=final_answer, cited_source_session=cited
    )


# --- normalization ---------------------------------------------------------


def test_normalize_numeric_strips_currency_and_commas():
    assert normalize_numeric("$1,234") == 1234.0
    assert normalize_numeric(" 6 ") == 6.0
    assert normalize_numeric("not a number") is None
    assert normalize_numeric(None) is None


def test_normalize_string_strips_quotes_case_whitespace():
    assert normalize_string(" 'Yes' ") == "yes"
    assert normalize_string('"NO"') == "no"


# --- is_correct --------------------------------------------------------


def test_int_answer_correct_and_incorrect():
    assert is_correct(INT_TASK, "6") is True
    assert is_correct(INT_TASK, "$6") is True
    assert is_correct(INT_TASK, "7") is False
    assert is_correct(INT_TASK, None) is False


def test_string_answer_correct_and_incorrect():
    assert is_correct(STRING_TASK, "Yes") is True
    assert is_correct(STRING_TASK, "'yes'") is True
    assert is_correct(STRING_TASK, "no") is False


# --- judge_session: non-trap -------------------------------------------


def test_dependent_task_correct():
    result = judge_session(INT_TASK, record("6"))
    assert result.correct is True
    assert result.category == "correct"


def test_dependent_task_wrong_has_no_interference_category():
    result = judge_session(INT_TASK, record("999"))
    assert result.correct is False
    assert result.category == "wrong"


# --- judge_session: trap -------------------------------------------------


def test_trap_task_correct():
    result = judge_session(TRAP_TASK, record("22"))
    assert result.correct is True
    assert result.category == "correct"


def test_trap_task_interference_confirmed_when_stale_answer_and_stale_citation():
    result = judge_session(TRAP_TASK, record("28", cited=17))
    assert result.correct is False
    assert result.category == "interference_confirmed"


def test_trap_task_interference_suspected_when_stale_answer_but_no_matching_citation():
    result = judge_session(TRAP_TASK, record("28", cited=None))
    assert result.category == "interference_suspected"

    result2 = judge_session(TRAP_TASK, record("28", cited=25))  # cited the *correct* source but still gave stale answer
    assert result2.category == "interference_suspected"


def test_trap_task_wrong_other_when_answer_matches_neither():
    result = judge_session(TRAP_TASK, record("999"))
    assert result.category == "wrong_other"


def test_trap_task_no_answer_is_wrong_other():
    result = judge_session(TRAP_TASK, record(None))
    assert result.correct is False
    assert result.category == "wrong_other"
