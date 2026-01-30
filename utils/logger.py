"""
Утилита для логирования интервью в JSON-формате.

Сохраняет историю диалога, внутренние мысли агентов
и итоговый отчёт в структурированном виде.
"""
import json
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path

from state import InterviewState, Turn


class InterviewLogger:
    """Класс для логирования хода интервью в JSON-формате."""
    
    def __init__(self, team_name: str = "Multi-Agent Interview Coach"):
        self.team_name = team_name
        self.log_data = {
            "team_name": team_name,
            "timestamp": datetime.now().isoformat(),
            "candidate": {},
            "turns": [],
            "internal_thoughts_log": [],
            "final_feedback": {}
        }
    
    def set_candidate_info(self, candidate: Dict[str, str]):
        """Устанавливает информацию о кандидате."""
        self.log_data["candidate"] = candidate
    
    def log_turn(self, turn_id: int, agent_message: str, 
                 user_message: str, internal_thoughts: List[Dict[str, str]]):
        """Логирует один ход диалога."""
        thoughts_str = ""
        for thought in internal_thoughts:
            thoughts_str += f"[{thought['from_agent']}]: {thought['content']} "
        
        turn = {
            "turn_id": turn_id,
            "agent_visible_message": agent_message,
            "user_message": user_message,
            "internal_thoughts": thoughts_str.strip()
        }
        
        self.log_data["turns"].append(turn)
        
        for thought in internal_thoughts:
            self.log_data["internal_thoughts_log"].append({
                "turn_id": turn_id,
                "from": thought["from_agent"],
                "to": thought["to_agent"],
                "content": thought["content"],
                "timestamp": thought.get("timestamp", datetime.now().isoformat())
            })
    
    def set_final_feedback(self, feedback: Dict[str, Any]):
        """Устанавливает итоговый отчёт."""
        self.log_data["final_feedback"] = feedback
    
    def save(self, filepath: str = None) -> str:
        """Сохраняет лог в JSON-файл и возвращает путь."""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"logs/interview_log_{timestamp}.json"
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.log_data, f, ensure_ascii=False, indent=2)
        
        return filepath
    
    def get_log_data(self) -> Dict[str, Any]:
        """Возвращает данные лога."""
        return self.log_data


def create_log_from_state(state: InterviewState, 
                          team_name: str = "Multi-Agent Interview Coach") -> Dict[str, Any]:
    """Создаёт структуру лога из состояния интервью."""
    log_data = {
        "team_name": team_name,
        "timestamp": datetime.now().isoformat(),
        "candidate": state["candidate"],
        "turns": [],
        "internal_thoughts_log": [],
        "final_feedback": state.get("final_feedback", {})
    }
    
    thoughts_by_turn = {}
    for thought in state.get("internal_thoughts", []):
        turn_id = len(thoughts_by_turn) + 1
        if turn_id not in thoughts_by_turn:
            thoughts_by_turn[turn_id] = []
        thoughts_by_turn[turn_id].append(thought)
    
    for turn in state.get("turns", []):
        turn_thoughts = thoughts_by_turn.get(turn["turn_id"], [])
        thoughts_str = " ".join([
            f"[{t['from_agent']}]: {t['content']}" 
            for t in turn_thoughts
        ])
        
        log_data["turns"].append({
            "turn_id": turn["turn_id"],
            "agent_visible_message": turn["agent_visible_message"],
            "user_message": turn["user_message"],
            "internal_thoughts": thoughts_str or turn.get("internal_thoughts", "")
        })
    
    for thought in state.get("internal_thoughts", []):
        log_data["internal_thoughts_log"].append({
            "from": thought["from_agent"],
            "to": thought["to_agent"],
            "content": thought["content"],
            "timestamp": thought.get("timestamp", "")
        })
    
    return log_data
