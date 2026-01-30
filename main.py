"""
Основной модуль мультиагентной системы для проведения технических интервью.

Multi-Agent Interview Coach — система из нескольких AI-агентов для
проведения технических интервью с возможностью скрытой рефлексии.
"""

import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from langchain_core.messages import HumanMessage, AIMessage

from state import create_initial_state, InterviewState, Turn
from graph import create_interview_graph
from utils.logger import InterviewLogger, create_log_from_state
from config import TEAM_NAME


class InterviewCoach:
    """Основной класс для проведения технических интервью."""
    
    def __init__(self, team_name: str = TEAM_NAME):
        self.team_name = team_name
        self.graph = create_interview_graph()
        self.state: Optional[InterviewState] = None
        self.logger = InterviewLogger(team_name)
        self.current_turn = 0
        
    def start_interview(
        self,
        candidate_name: str,
        position: str,
        grade: str,
        experience: str,
        difficulty_mode: str = "adaptive",
        initial_difficulty_level: Optional[int] = None,
    ) -> str:
        """Начинает новое интервью и возвращает первое сообщение интервьюера."""
        self.state = create_initial_state(
            candidate_name=candidate_name,
            position=position,
            grade=grade,
            experience=experience,
            difficulty_mode=difficulty_mode,
            initial_difficulty_level=initial_difficulty_level,
        )
        
        self.logger.set_candidate_info({
            "name": candidate_name,
            "position": position,
            "grade": grade,
            "experience": experience
        })

        result = self.graph.invoke(self.state)
        self.state = result

        self.state["messages"] = self.state.get("messages", [])[-3:]
        
        ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
        first_message = ai_messages[-1].content if ai_messages else "Привет! Давайте начнем интервью."
        
        return first_message
    
    def process_response(self, user_message: str) -> str:
        """Обрабатывает ответ кандидата и возвращает следующее сообщение."""
        if self.state is None:
            return "Ошибка: интервью не начато. Вызовите start_interview() сначала."
        
        self.current_turn += 1
        
        ai_messages = [m for m in self.state["messages"] if isinstance(m, AIMessage)]
        last_agent_message = ai_messages[-1].content if ai_messages else ""
        
        self.state["messages"] = self.state["messages"] + [HumanMessage(content=user_message)]
        self.state["current_turn"] = self.current_turn

        self.state["messages"] = self.state.get("messages", [])[-3:]
        
        result = self.graph.invoke(self.state)
        self.state = result

        self.state["messages"] = self.state.get("messages", [])[-3:]
        
        new_thoughts = result.get("internal_thoughts", [])
        thoughts_for_turn = new_thoughts[-2:] if len(new_thoughts) >= 2 else new_thoughts
        
        self.logger.log_turn(
            turn_id=self.current_turn,
            agent_message=last_agent_message,
            user_message=user_message,
            internal_thoughts=[
                {
                    "from_agent": t["from_agent"],
                    "to_agent": t["to_agent"],
                    "content": t["content"],
                    "timestamp": t.get("timestamp", "")
                }
                for t in thoughts_for_turn
            ]
        )
        
        turn = Turn(
            turn_id=self.current_turn,
            agent_visible_message=last_agent_message,
            user_message=user_message,
            internal_thoughts=" ".join([
                f"[{t['from_agent']}]: {t['content']}" 
                for t in thoughts_for_turn
            ])
        )
        self.state["turns"] = self.state.get("turns", []) + [turn]
        
        ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
        response = ai_messages[-1].content if ai_messages else "Продолжим интервью."
        
        if result.get("interview_finished"):
            self.logger.set_final_feedback(result.get("final_feedback", {}))
        
        return response
    
    def end_interview(self) -> str:
        """Принудительно завершает интервью и генерирует отчёт."""
        return self.process_response("Стоп игра. Давай фидбэк.")
    
    def save_log(self, filepath: str = None) -> str:
        """Сохраняет лог интервью в JSON-файл."""
        return self.logger.save(filepath)
    
    def get_state(self) -> InterviewState:
        """Возвращает текущее состояние интервью."""
        return self.state


def run_scenario():
    """Запускает тестовый сценарий интервью."""
    print("=" * 60)
    print("MULTI-AGENT INTERVIEW COACH")
    print("=" * 60)
    print()
    
    coach = InterviewCoach()
    
    candidate_info = {
        "name": "Алекс",
        "position": "Backend Developer",
        "grade": "Junior",
        "experience": "Пет-проекты на Django, немного SQL"
    }
    
    print(f"Кандидат: {candidate_info['name']}")
    print(f"Позиция: {candidate_info['position']}")
    print(f"Грейд: {candidate_info['grade']}")
    print(f"Опыт: {candidate_info['experience']}")
    print()
    print("-" * 60)
    
    first_message = coach.start_interview(
        candidate_name=candidate_info['name'],
        position=candidate_info['position'],
        grade=candidate_info['grade'],
        experience=candidate_info['experience']
    )
    print(f"\nИнтервьюер: {first_message}")
    scenario = [ 
        { 
            "step": "Ход 1 (Приветствие)",
            "response": "Привет. Я Алекс, претендую на позицию Junior Backend Developer. Знаю Python, SQL и Git." 
        }, 
        {
            "step": "Ход 2 (Проверка знаний)",
            "response": "Список в Python - это изменяемая упорядоченная коллекция элементов. Основные методы: append() для добавления элемента в конец, pop() для удаления последнего элемента, insert() для вставки по индексу. Списки поддерживают срезы и итерацию. Словарь - это неупорядоченная коллекция пар ключ-значение, где ключи должны быть хешируемыми." 
        }, 
        { 
            "step": "Ход 3 (Hallucination Test)", 
            "response": "Честно говоря, я читал на Хабре, что в Python 4.0 циклы for уберут и заменят на нейронные связи, поэтому я их не учу."
        }, 
        { 
            "step": "Ход 4 (Role Reversal)",
            "response": "Слушайте, а какие задачи вообще будут на испытательном сроке? Вы используете микросервисы?"
        }, 
        { 
            "step": "Ход 5 (Завершение)",
            "response": "Стоп игра. Давай фидбэк."
        }
    ]

    
    for step in scenario:
        print(f"\n{step['step']}")
        print(f"Кандидат: {step['response']}")
        
        response = coach.process_response(step['response'])
        print(f"\nИнтервьюер: {response}")
        print("-" * 60)
    
    log_path = coach.save_log()
    print(f"\nЛог сохранен: {log_path}")
    
    return coach


def interactive_mode():
    """Интерактивный режим интервью."""
    print("=" * 60)
    print("MULTI-AGENT INTERVIEW COACH - Интерактивный режим")
    print("=" * 60)
    print()
    
    print("Введите данные кандидата:")
    name = input("Имя: ").strip() or "Кандидат"
    position = input("Позиция: ").strip() or "Backend Developer"
    grade = input("Грейд (Junior/Middle/Senior): ").strip() or "Junior"
    experience = input("Опыт: ").strip() or "Начинающий разработчик"
    
    coach = InterviewCoach()
    
    first_message = coach.start_interview(
        candidate_name=name,
        position=position,
        grade=grade,
        experience=experience
    )
    
    print(f"\nИнтервьюер: {first_message}")
    print("\n(Введите 'стоп' или 'выход' для завершения)")
    print("-" * 60)
    
    while True:
        user_input = input("\nВы: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() in ['выход', 'exit', 'quit']:
            print("\nЗавершение без сохранения...")
            break
        
        response = coach.process_response(user_input)
        print(f"\nИнтервьюер: {response}")
        
        if coach.state.get("interview_finished"):
            log_path = coach.save_log()
            print(f"\nЛог сохранен: {log_path}")
            break
    
    return coach


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_mode()
    else:
        run_scenario()
