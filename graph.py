"""
Граф LangGraph для оркестрации мультиагентной системы интервью.

Определяет узлы графа и логику переходов между агентами.
"""
import json
from typing import Literal, Dict, Any
from datetime import datetime

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage

from state import InterviewState, Turn, create_initial_state
from agents.agents import (
    InterviewerAgent,
    MentorAgent,
    FactCheckerAgent,
    FeedbackGeneratorAgent,
    create_thought
)
from config import MAX_TURNS


interviewer = InterviewerAgent()
mentor = MentorAgent()
fact_checker = FactCheckerAgent()
feedback_generator = FeedbackGeneratorAgent()


def interviewer_node(state: InterviewState) -> Dict[str, Any]:
    """Узел интервьюера: генерирует вопрос или ответ для кандидата."""
    mentor_recommendation = None
    if state["internal_thoughts"]:
        last_thought = state["internal_thoughts"][-1]
        if last_thought["to_agent"] == "Interviewer_Agent":
            mentor_recommendation = last_thought["content"]
    
    response = interviewer.generate_question(state, mentor_recommendation)
    new_message = AIMessage(content=response)
    
    return {
        "messages": [new_message]
    }


def mentor_node(state: InterviewState) -> Dict[str, Any]:
    """Узел ментора: анализирует ответ кандидата и даёт рекомендации."""
    user_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    if not user_messages:
        return {}
    
    last_user_message = user_messages[-1].content
    analysis = mentor.analyze(state, last_user_message)

    difficulty_mode = (state.get("difficulty_mode") or "adaptive").lower()
    difficulty_level = int(state.get("difficulty_level", 2) or 2)
    difficulty_level = max(1, min(10, difficulty_level))
    correct_streak = int(state.get("correct_streak", 0) or 0)
    difficulty_history = list(state.get("difficulty_history") or [difficulty_level])
    
    red_flags = []
    mentor_flags = analysis.get("red_flags") or []
    confidence_level = str(analysis.get("confidence_level") or "").lower()
    correctness_score = analysis.get("correctness_score")
    is_correct = analysis.get("is_correct")
    is_hallucination = bool(analysis.get("is_hallucination"))

    has_hallucination_hint = False
    for flag in mentor_flags:
        flag_lower = str(flag).lower()
        if any(token in flag_lower for token in ["галлю", "hallucin", "ложн", "false fact", "factually"]):
            has_hallucination_hint = True
            break

    is_high_confidence = confidence_level in {"high", "высок", "высокий", "высокая"}
    is_low_score = isinstance(correctness_score, (int, float)) and correctness_score <= 40
    is_confidently_wrong = is_high_confidence and (is_correct is False or is_low_score)

    should_fact_check = (
        is_hallucination
        or has_hallucination_hint
        or is_confidently_wrong
        or "python 4" in last_user_message.lower()
    )

    if should_fact_check:
        fact_result = fact_checker.check(last_user_message)
        if fact_result.get("is_true") is False:
            explanation = fact_result.get("explanation", "")
            correct_info = fact_result.get("correct_info", "")
            red_flags.append(
                f"ГАЛЛЮЦИНАЦИЯ/ЛОЖНЫЙ ФАКТ: {explanation}. Правильная информация: {correct_info}".strip()
            )

            thought = create_thought(
                from_agent="FactChecker_Agent",
                to_agent="Interviewer_Agent",
                content=f"ALERT: Кандидат уверенно сообщает ложные факты. {explanation}. "
                        f"Правильная информация: {correct_info}. "
                        f"Это критическая ошибка знаний. Пометь как 'red flag'."
            )
            state["internal_thoughts"].append(thought)

    topics_covered = list(state.get("topics_covered") or [])
    topic_detected = (analysis.get("topic_detected") or "").strip()
    if topic_detected and topic_detected not in topics_covered:
        topics_covered.append(topic_detected)

    if difficulty_mode == "adaptive":
        old_difficulty = difficulty_level
        correctness_score = analysis.get("correctness_score")
        is_correct = analysis.get("is_correct")
        recommendation = (analysis.get("difficulty_recommendation") or "maintain").lower()

        if is_correct is True and isinstance(correctness_score, (int, float)) and correctness_score >= 80:
            correct_streak += 1
        else:
            correct_streak = 0

        if recommendation == "increase" or correct_streak >= 2:
            difficulty_level = min(10, difficulty_level + 1)
            correct_streak = 0
        elif recommendation == "decrease":
            difficulty_level = max(1, difficulty_level - 1)
        elif isinstance(correctness_score, (int, float)) and correctness_score <= 40:
            difficulty_level = max(1, difficulty_level - 1)

        if difficulty_level != old_difficulty:
            difficulty_history.append(difficulty_level)
    
    recommendation = analysis.get("recommendation", "Продолжить интервью")
    if analysis.get("suggested_action") == "challenge" or red_flags:
        recommendation = f"ВАЖНО: Кандидат допустил ошибку. {recommendation}. Вежливо укажи на неточность."
    
    thought = create_thought(
        from_agent="Mentor_Agent",
        to_agent="Interviewer_Agent",
        content=f"Анализ: {analysis.get('analysis', '')}. "
                f"Уровень уверенности: {analysis.get('confidence_level', 'unknown')}. "
                f"Рекомендация: {recommendation}"
    )
    
    confirmed_skills = list(state["confirmed_skills"])
    knowledge_gaps = list(state["knowledge_gaps"])
    
    if analysis.get("is_correct"):
        topic = analysis.get("analysis", "")[:50]
        if topic and topic not in confirmed_skills:
            confirmed_skills.append(topic)
    elif analysis.get("is_correct") is False:
        gap = {
            "topic": analysis.get("analysis", "")[:50],
            "question": "",
            "correct_answer": ""
        }
        knowledge_gaps.append(gap)
    
    return {
        "internal_thoughts": state["internal_thoughts"] + [thought],
        "red_flags": state["red_flags"] + red_flags,
        "topics_covered": topics_covered,
        "confirmed_skills": confirmed_skills,
        "knowledge_gaps": knowledge_gaps,
        "difficulty_level": difficulty_level,
        "correct_streak": correct_streak,
        "difficulty_mode": difficulty_mode,
        "difficulty_history": difficulty_history
    }


def feedback_node(state: InterviewState) -> Dict[str, Any]:
    """Узел генерации итогового отчёта."""
    feedback = feedback_generator.generate(state)
    
    report = f"""
**ИТОГОВЫЙ ОТЧЕТ ПО ИНТЕРВЬЮ**

**Кандидат:** {state['candidate']['name']}
**Позиция:** {state['candidate']['position']}

---

## Вердикт

| Параметр | Значение |
|----------|----------|
| **Оценка уровня** | {feedback['verdict']['grade']} |
| **Рекомендация** | {feedback['verdict']['hiring_recommendation']} |
| **Уверенность** | {feedback['verdict']['confidence_score']}% |

---

## Технический обзор

### Подтвержденные навыки:
"""
    for skill in feedback['technical_review']['confirmed_skills']:
        report += f"- {skill}\n"
    
    report += "\n### Пробелы в знаниях:\n"
    for gap in feedback['technical_review']['knowledge_gaps']:
        if isinstance(gap, dict):
            report += f"- **{gap.get('topic', 'N/A')}**\n"
            if gap.get('correct_answer'):
                report += f"  - Правильный ответ: {gap['correct_answer']}\n"
        else:
            report += f"- {gap}\n"
    
    report += f"""
---

## Soft Skills

| Навык | Оценка |
|-------|--------|
| Ясность изложения | {feedback['soft_skills']['clarity']}/10 |
| Честность | {feedback['soft_skills']['honesty']}/10 |
| Вовлеченность | {feedback['soft_skills']['engagement']}/10 |

**Комментарии:** {feedback['soft_skills']['comments']}

---

## Рекомендации по развитию

### Темы для изучения:
"""
    for topic in feedback['roadmap']['topics_to_improve']:
        report += f"- {topic}\n"
    
    report += "\n### Рекомендуемые ресурсы:\n"
    for resource in feedback['roadmap']['resources']:
        report += f"- {resource}\n"
    
    report += f"""
---

## Резюме

{feedback['summary']}
"""
    
    new_message = AIMessage(content=report)
    
    return {
        "messages": [new_message],
        "final_feedback": feedback,
        "interview_finished": True
    }


def should_continue(state: InterviewState) -> Literal["mentor", "feedback", "end"]:
    """Определяет следующий узел после интервьюера."""
    if state["interview_finished"]:
        return "end"
    
    user_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    if user_messages:
        last_message = user_messages[-1].content.lower()
        if any(cmd in last_message for cmd in ["стоп", "stop", "завершить", "фидбэк", "feedback"]):
            return "feedback"
    
    if state["current_turn"] >= MAX_TURNS:
        return "feedback"
    
    return "mentor"


def should_interview(state: InterviewState) -> Literal["interviewer", "end"]:
    """Определяет, продолжать ли интервью после анализа ментора."""
    if state["interview_finished"]:
        return "end"
    return "interviewer"


def create_interview_graph() -> StateGraph:
    """Создаёт и компилирует граф для проведения интервью."""
    workflow = StateGraph(InterviewState)
    
    workflow.add_node("interviewer", interviewer_node)
    workflow.add_node("mentor", mentor_node)
    workflow.add_node("feedback", feedback_node)
    
    workflow.set_entry_point("interviewer")
    
    workflow.add_conditional_edges(
        "interviewer",
        should_continue,
        {
            "mentor": "mentor",
            "feedback": "feedback",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "mentor",
        should_interview,
        {
            "interviewer": "interviewer",
            "end": END
        }
    )
    
    workflow.add_edge("feedback", END)
    
    return workflow.compile()
