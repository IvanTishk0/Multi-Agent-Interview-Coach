"""
Определения состояния для системы интервью.

Содержит TypedDict-классы для типизации данных,
передаваемых между агентами.
"""
from typing import TypedDict, List, Dict, Any, Annotated, Optional
from langchain_core.messages import BaseMessage
import operator


class InternalThought(TypedDict):
    """Внутренняя мысль агента, невидимая для кандидата."""
    from_agent: str
    to_agent: str
    content: str
    timestamp: str


class Turn(TypedDict):
    """Один ход диалога интервью."""
    turn_id: int
    agent_visible_message: str
    user_message: str
    internal_thoughts: str


class CandidateInfo(TypedDict):
    """Информация о кандидате."""
    name: str
    position: str
    grade: str
    experience: str


class InterviewState(TypedDict):
    """
    Главный объект состояния интервью.
    
    Передаётся между агентами и содержит всю информацию
    о текущем состоянии интервью.
    """
    messages: Annotated[List[BaseMessage], operator.add]
    candidate: CandidateInfo
    turns: List[Turn]
    internal_thoughts: List[InternalThought]
    current_turn: int
    topics_covered: List[str]
    confirmed_skills: List[str]
    knowledge_gaps: List[Dict[str, str]]
    red_flags: List[str]
    difficulty_level: int
    correct_streak: int
    difficulty_mode: str
    difficulty_history: List[int]
    interview_finished: bool
    final_feedback: Dict[str, Any]


def _grade_to_initial_difficulty(grade: str) -> int:
    grade_lower = (grade or "").lower()
    if "senior" in grade_lower:
        return 8
    if "middle" in grade_lower:
        return 6
    if "junior+" in grade_lower or "junior plus" in grade_lower:
        return 5
    if "junior" in grade_lower:
        return 3
    return 2


def create_initial_state(
    *,
    candidate_name: str = None,
    name: str = None,
    position: str = None,
    pos: str = None,
    grade: Optional[str] = None,
    experience: str = None,
    exp: str = None,
    difficulty_mode: str = "adaptive",
    initial_difficulty_level: Optional[int] = None,
) -> InterviewState:
    """Создаёт начальное состояние интервью."""
    resolved_name = candidate_name or name
    resolved_pos = position or pos
    resolved_exp = experience or exp

    if not resolved_name or not resolved_pos or resolved_exp is None:
        raise ValueError("create_initial_state requires candidate_name/name, position/pos, experience/exp")

    if not grade:
        raise ValueError("create_initial_state requires grade")

    if initial_difficulty_level is None:
        difficulty_level = _grade_to_initial_difficulty(grade)
    else:
        difficulty_level = max(1, min(10, int(initial_difficulty_level)))

    return InterviewState(
        messages=[],
        candidate=CandidateInfo(
            name=resolved_name,
            position=resolved_pos,
            grade=grade,
            experience=resolved_exp
        ),
        turns=[],
        internal_thoughts=[],
        current_turn=0,
        topics_covered=[],
        confirmed_skills=[],
        knowledge_gaps=[],
        red_flags=[],
        difficulty_level=difficulty_level,
        correct_streak=0,
        difficulty_mode=difficulty_mode,
        difficulty_history=[difficulty_level],
        interview_finished=False,
        final_feedback={}
    )
