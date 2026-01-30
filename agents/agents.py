"""
Модуль агентов для системы интервью.

Содержит классы агентов: InterviewerAgent, MentorAgent,
FactCheckerAgent, FeedbackGeneratorAgent.
"""
import os
import json
from datetime import datetime
from typing import Dict, Any
from openai import OpenAI

from config import (
    LLM_MODEL, LLM_TEMPERATURE,
    GROQ_API_KEY, GROQ_BASE_URL,
    INTERVIEWER_SYSTEM_PROMPT, MENTOR_SYSTEM_PROMPT,
    FACT_CHECKER_PROMPT, FEEDBACK_GENERATOR_PROMPT,
    TOPICS_BY_GRADE,
    ROLE_LABELS,
    TOPICS_BY_ROLE_AND_DIFFICULTY,
    TOPICS_BY_ROLE_AND_GRADE,
    detect_role_from_position
)
from state import InterviewState, InternalThought, Turn

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url=GROQ_BASE_URL
)


class InterviewerAgent:
    """Агент-интервьюер, ведущий диалог с кандидатом."""
    
    def __init__(self):
        self.name = "Interviewer_Agent"
        self.model = LLM_MODEL
        
    def generate_question(self, state: InterviewState, mentor_rec: str = None) -> str:
        """Генерирует вопрос или ответ интервьюера."""
        cand = state["candidate"]
        turn = state["current_turn"]

        detected_role = detect_role_from_position(cand.get("position", ""), default="backend")
        
        sys_prompt = INTERVIEWER_SYSTEM_PROMPT.format(
            candidate_name=cand["name"],
            position=cand["position"],
            grade=cand["grade"],
            experience=cand["experience"],
            turn_number=turn
        )
        sys_prompt += f"\nПрофессиональная роль (определена системой): {ROLE_LABELS.get(detected_role, detected_role)}\n"
        
        difficulty_level = int(state.get("difficulty_level", 2) or 2)
        difficulty_level = max(1, min(10, difficulty_level))
        sys_prompt += f"Текущий уровень сложности (1-10): {difficulty_level}\n"
        
        msgs = [{"role": "system", "content": sys_prompt}]

        recent_messages = state["messages"][-3:]
        for msg in recent_messages:
            message_role = "assistant" if msg.type == "ai" else "user"
            msgs.append({"role": message_role, "content": msg.content})
        
        if mentor_rec:
            msgs.append({
                "role": "system",
                "content": f"[INTERNAL - MENTOR]: {mentor_rec}"
            })
        
        topics = TOPICS_BY_ROLE_AND_DIFFICULTY.get(detected_role, {}).get(difficulty_level)
        if not topics:
            topics = TOPICS_BY_ROLE_AND_GRADE.get(detected_role, {}).get(
                cand["grade"],
                TOPICS_BY_GRADE.get(cand["grade"], TOPICS_BY_GRADE["Junior"])
            )
        covered = state["topics_covered"]
        remaining = [t for t in topics if t not in covered]
        
        if remaining and turn > 0:
            msgs.append({
                "role": "system",
                "content": f"Available topics: {', '.join(remaining[:3])}"
            })
        
        resp = client.chat.completions.create(
            model=self.model,
            messages=msgs,
            temperature=LLM_TEMPERATURE
        )
        
        return resp.choices[0].message.content
    
    def respond_to_question(self, state: InterviewState, question: str) -> str:
        """Отвечает на вопрос кандидата."""
        cand = state["candidate"]
        
        sys_prompt = f"""You are an interviewer. Candidate asked you a question. 
Answer briefly and professionally, then continue the interview.

Position: {cand["position"]}
Question: {question}

Answer and ask next technical question."""
        
        resp = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": sys_prompt}],
            temperature=LLM_TEMPERATURE
        )
        
        return resp.choices[0].message.content


class MentorAgent:
    """Агент-ментор, анализирующий ответы кандидата."""
    
    def __init__(self):
        self.name = "Mentor_Agent"
        self.model = LLM_MODEL
        
    def analyze(self, state: InterviewState, user_msg: str) -> Dict[str, Any]:
        """Анализирует ответ кандидата и возвращает рекомендации."""
        cand = state["candidate"]
        
        ctx = f"""
            Candidate info:
            - Name: {cand["name"]}
            - Position: {cand["position"]}
            - Grade: {cand["grade"]}
            - Experience: {cand["experience"]}
            
            Dialog history:
        """

        recent_messages = state["messages"][-3:]
        for msg in recent_messages:
            role = "Interviewer" if msg.type == "ai" else "Candidate"
            ctx += f"\n{role}: {msg.content}"

        ctx += f"\n\nLast response: {user_msg}"
        
        msgs = [
            {"role": "system", "content": MENTOR_SYSTEM_PROMPT},
            {"role": "user", "content": ctx}
        ]
        
        resp = client.chat.completions.create(
            model=self.model,
            messages=msgs,
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        try:
            result = json.loads(resp.choices[0].message.content)
        except json.JSONDecodeError:
            result = {
                "analysis": "Failed to analyze",
                "is_correct": None,
                "confidence_level": "unknown",
                "red_flags": [],
                "recommendation": "Continue interview",
                "suggested_action": "move_on"
            }
        
        return result


class FactCheckerAgent:
    """Агент для проверки фактов в ответах кандидата."""
    
    def __init__(self):
        self.name = "FactChecker_Agent"
        self.model = LLM_MODEL
        
    def check(self, statement: str) -> Dict[str, Any]:
        """Проверяет утверждение на достоверность."""
        prompt = FACT_CHECKER_PROMPT.format(statement=statement)
        
        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an IT fact-checking expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        try:
            result = json.loads(resp.choices[0].message.content)
        except json.JSONDecodeError:
            result = {
                "is_true": None,
                "explanation": "Could not verify",
                "correct_info": ""
            }
        
        return result


class FeedbackGeneratorAgent:
    """Агент для генерации итогового отчёта по интервью."""
    
    def __init__(self):
        self.name = "FeedbackGenerator_Agent"
        self.model = LLM_MODEL
        
    def generate(self, state: InterviewState) -> Dict[str, Any]:
        """Генерирует итоговый отчёт на основе истории интервью."""
        cand = state["candidate"]
        
        history = ""
        for turn in state["turns"]:
            history += f"\n--- Turn {turn['turn_id']} ---"
            history += f"\nInterviewer: {turn['agent_visible_message']}"
            history += f"\nCandidate: {turn['user_message']}"
            if turn.get('internal_thoughts'):
                history += f"\n[Internal]: {turn['internal_thoughts']}"
        
        if state["red_flags"]:
            history += f"\n\n--- Issues found ---"
            for flag in state["red_flags"]:
                history += f"\n- {flag}"
        
        prompt = FEEDBACK_GENERATOR_PROMPT.format(
            interview_history=history,
            candidate_name=cand["name"],
            position=cand["position"],
            grade=cand["grade"],
            experience=cand["experience"]
        )
        
        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a candidate evaluation expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            response_format={"type": "json_object"}
        )
        
        try:
            result = json.loads(resp.choices[0].message.content)
        except json.JSONDecodeError:
            result = {
                "verdict": {
                    "grade": "Unknown",
                    "hiring_recommendation": "No Hire",
                    "confidence_score": 0
                },
                "technical_review": {
                    "confirmed_skills": state["confirmed_skills"],
                    "knowledge_gaps": state["knowledge_gaps"]
                },
                "soft_skills": {
                    "clarity": 5,
                    "honesty": 5,
                    "engagement": 5,
                    "comments": "Could not evaluate"
                },
                "roadmap": {
                    "topics_to_improve": [],
                    "resources": []
                },
                "summary": "Failed to generate report"
            }
        
        return result


def create_thought(from_agent: str, to_agent: str, content: str) -> InternalThought:
    """Создаёт запись внутренней мысли агента."""
    return InternalThought(
        from_agent=from_agent,
        to_agent=to_agent,
        content=content,
        timestamp=datetime.now().isoformat()
    )
