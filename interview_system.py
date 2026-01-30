"""
Мультиагентная система для проведения технических интервью.
"""

import os
import json
import re
import sys
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from openai import OpenAI
from log_adapter import LogAdapter

from config import (
    GROQ_API_KEY,
    GROQ_BASE_URL,
    LLM_MODEL,
    ROLE_LABELS,
    TOPICS_BY_ROLE_AND_DIFFICULTY,
    detect_role_from_position,
)

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url=GROQ_BASE_URL
)


def call_llm(prompt: str, temperature: float = 0.7) -> str:
    """Вызывает LLM и возвращает текст ответа."""
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )
    return resp.choices[0].message.content


def extract_json(text: str) -> Dict:
    """Извлекает JSON из текста, даже если он обёрнут в markdown."""
    text = re.sub(r'^```json\s*', '', text.strip())
    text = re.sub(r'^```\s*', '', text)
    text = re.sub(r'\s*```$', '', text)

    json_match = re.search(r'\{[\s\S]*\}', text)
    if json_match:
        try:
            return json.loads(json_match.group())
        except:
            pass
    
    try:
        return json.loads(text)
    except:
        return {}


def call_llm_with_json_retry(prompt: str, max_retries: int = 3, temperature: float = 0.7) -> Dict:
    """
    Retry логика для получения валидного JSON.
    
    Вызывает LLM и пытается получить валидный JSON.
    При неудаче переформулирует запрос и пробует снова.
    """
    for attempt in range(max_retries):
        try:
            current_temp = max(0.3, temperature - (attempt * 0.2))
            
            if attempt > 0:
                retry_prompt = prompt + f"""

КРИТИЧЕСКИ ВАЖНО (попытка {attempt + 1}):
- Верни ТОЛЬКО валидный JSON
- НЕ добавляй никакого текста до или после JSON
- НЕ используй ```json или другие обёртки
- Начни ответ сразу с открывающей фигурной скобки {{
"""
                resp = call_llm(retry_prompt, current_temp)
            else:
                resp = call_llm(prompt, current_temp)
            
            result = extract_json(resp)
            
            if result:
                return result

            if attempt < max_retries - 1:
                time.sleep(0.5)
                
        except Exception as e:
            print(f"[DEBUG] Ошибка при вызове LLM (попытка {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
    
    return {}


@dataclass
class InternalThought:
    """Внутренняя мысль агента, невидимая для кандидата."""
    from_agent: str
    to_agent: str
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class Turn:
    """Один ход диалога интервью."""
    turn_id: int
    agent_visible_message: str
    user_message: str
    internal_thoughts: str


class InterviewerAgent:
    """Агент-интервьюер, ведущий диалог с кандидатом."""
    
    def __init__(self):
        self.name = "Interviewer"

        self.topics_by_difficulty = {
            1: ["что такое переменная", "базовые типы данных (int, str, bool)", "print и input"],
            2: ["списки и кортежи", "словари", "условные операторы if/else"],
            3: ["циклы for и while", "функции и return", "работа со строками"],
            4: ["аргументы функций (*args, **kwargs)", "обработка исключений try/except", "работа с файлами"],
            5: ["ООП: классы и объекты", "наследование", "инкапсуляция и полиморфизм"],
            6: ["декораторы", "генераторы и итераторы", "контекстные менеджеры"],
            7: ["многопоточность (threading)", "асинхронность async/await", "работа с API"],
            8: ["паттерны проектирования", "SOLID принципы", "тестирование (pytest, unittest)"],
            9: ["архитектура приложений", "микросервисы", "Docker и контейнеризация"],
            10: ["распределённые системы", "масштабирование и оптимизация", "системный дизайн"]
        }
    
    def generate_response(self, cand: Dict[str, str], history: List[Dict], 
                         mentor_advice: str = "", turn: int = 0,
                         difficulty: int = 2, correct_streak: int = 0,
                         topics_covered: List[str] = None,
                         edge_case: str = None) -> str:
        
        role = detect_role_from_position(cand.get("position", ""), default="backend")
        role_topics_map = TOPICS_BY_ROLE_AND_DIFFICULTY.get(role, {})
        topics = role_topics_map.get(difficulty) or self.topics_by_difficulty.get(difficulty, self.topics_by_difficulty[2])

        difficulty_desc = {
            1: "НАЧАЛЬНЫЙ (основы программирования)",
            2: "БАЗОВЫЙ (структуры данных)",
            3: "JUNIOR- (базовый Python)",
            4: "JUNIOR (функции и исключения)",
            5: "JUNIOR+ (ООП)",
            6: "MIDDLE- (продвинутый Python)",
            7: "MIDDLE (асинхронность и API)",
            8: "MIDDLE+ (архитектура кода)",
            9: "SENIOR- (инфраструктура)",
            10: "SENIOR (системный дизайн)"
        }
        
        topics_info = ""
        if topics_covered:
            topics_info = f"""
УЖЕ ЗАТРОНУТЫЕ ТЕМЫ (НЕ ЗАДАВАЙ ВОПРОСЫ ПО НИМ):
{', '.join(topics_covered)}
"""
        
        edge_case_instruction = ""
        if edge_case == "silence":
            edge_case_instruction = """
ОСОБАЯ СИТУАЦИЯ: Кандидат молчит или дал пустой ответ.
- Мягко подбодри кандидата
- Предложи подсказку или переформулируй вопрос проще
- Скажи что-то вроде: "Не переживайте, давайте попробуем по-другому..."
"""
        elif edge_case == "dont_know":
            edge_case_instruction = """
ОСОБАЯ СИТУАЦИЯ: Кандидат сказал "не знаю" или признал незнание.
- Похвали за честность
- Кратко объясни правильный ответ (1-2 предложения)
- Перейди к более простому вопросу или другой теме
- Скажи что-то вроде: "Ничего страшного, это нормально. Правильный ответ: ... Давайте попробуем другую тему."
"""
        elif edge_case == "rude":
            edge_case_instruction = """
ОСОБАЯ СИТУАЦИЯ: Кандидат грубит или ведёт себя непрофессионально.
- Сохраняй спокойствие и профессионализм
- Не отвечай грубостью на грубость
- Мягко напомни о формате интервью
- Если продолжается — предложи завершить интервью
- Скажи что-то вроде: "Давайте вернёмся к профессиональному формату беседы."
"""
        elif edge_case == "troll":
            edge_case_instruction = """
ОСОБАЯ СИТУАЦИЯ: Кандидат троллит или даёт абсурдные ответы.
- Не поддавайся на провокации
- Вежливо игнорируй абсурд
- Переформулируй вопрос или задай новый
- Скажи что-то вроде: "Интересная точка зрения. Давайте вернёмся к техническому вопросу..."
"""
        
        prompt = f"""Ты опытный технический интервьюер в IT-компании. Проводишь собеседование.

ИНФОРМАЦИЯ О КАНДИДАТЕ:
- Имя: {cand['name']}
- Позиция: {cand['position']}
- Грейд: {cand['grade']}
- Опыт: {cand['experience']}

 ПРОФЕССИОНАЛЬНАЯ РОЛЬ (определена системой): {ROLE_LABELS.get(role, role)}

ТЕКУЩИЙ УРОВЕНЬ СЛОЖНОСТИ: {difficulty}/10 ({difficulty_desc.get(difficulty, 'N/A')})
ПРАВИЛЬНЫХ ОТВЕТОВ ПОДРЯД: {correct_streak}

ТЕМЫ ДЛЯ ВОПРОСОВ НА ЭТОМ УРОВНЕ: {', '.join(topics)}
{topics_info}
{edge_case_instruction}

ПРАВИЛА:
1. Задавай по ОДНОМУ техническому вопросу за раз
2. Вопрос должен соответствовать текущему уровню сложности ({difficulty}/10)
3. Если кандидат задает встречный вопрос - ОБЯЗАТЕЛЬНО ответь на него кратко и профессионально, затем продолжи интервью
4. НЕ соглашайся с явно неверными утверждениями
5. Если кандидат говорит что-то странное про Python 4.0 или другие несуществующие технологии - вежливо укажи на ошибку и дай правильную информацию
6. Будь профессионален и дружелюбен
7. Если кандидат уходит от темы интервью - вежливо верни беседу к техническим вопросам
8. НЕ задавай вопросы по темам, которые уже были затронуты
9. При ответе "не знаю" - кратко объясни правильный ответ и перейди к другой теме

ТЕКУЩИЙ ХОД: {turn}
"""
        
        if mentor_advice:
            prompt += f"\n[ВНУТРЕННЯЯ РЕКОМЕНДАЦИЯ ОТ OBSERVER]: {mentor_advice}"
        
        prompt += "\n\nИСТОРИЯ ДИАЛОГА:\n"
        for h in history:
            role = "Интервьюер" if h["role"] == "assistant" else "Кандидат"
            prompt += f"{role}: {h['content']}\n"
        
        prompt += "\nТвой ответ как интервьюера:"
        
        return call_llm(prompt)


class MentorAgent:
    """
    Агент-ментор (Observer), анализирующий ответы кандидата в реальном времени.
    Работает "за кулисами", невидимо для кандидата.
    """
    
    def __init__(self):
        self.name = "Observer"

        self.few_shot_examples = """
ПРИМЕРЫ АНАЛИЗА:

Пример 1 - Хороший ответ:
Вопрос: "Что такое декоратор в Python?"
Ответ кандидата: "Декоратор — это функция, которая принимает другую функцию и расширяет её поведение без явного изменения кода. Используется синтаксис @decorator_name над функцией."
Анализ:
{
    "analysis": "Кандидат дал точное и полное определение декоратора, упомянул синтаксис. Демонстрирует понимание концепции.",
    "is_correct": true,
    "correctness_score": 95,
    "confidence_level": "high",
    "red_flags": [],
    "is_hallucination": false,
    "is_question_from_candidate": false,
    "is_off_topic": false,
    "topic_detected": "декораторы",
    "difficulty_recommendation": "increase",
    "recommendation": "Можно углубиться в тему — спросить про декораторы с аргументами или functools.wraps",
    "suggested_action": "challenge"
}

Пример 2 - Слабый ответ:
Вопрос: "Чем отличается list от tuple?"
Ответ кандидата: "Ну... list это список, а tuple тоже список, но немного другой..."
Анализ:
{
    "analysis": "Кандидат не смог чётко сформулировать различия. Ответ размытый и неуверенный.",
    "is_correct": false,
    "correctness_score": 20,
    "confidence_level": "low",
    "red_flags": ["Не знает ключевое различие: изменяемость"],
    "is_hallucination": false,
    "is_question_from_candidate": false,
    "is_off_topic": false,
    "topic_detected": "структуры данных",
    "difficulty_recommendation": "decrease",
    "recommendation": "Объяснить различие и задать более простой вопрос",
    "suggested_action": "simplify"
}

Пример 3 - Галлюцинация:
Вопрос: "Какие новые фичи в Python?"
Ответ кандидата: "В Python 4.0 добавили нативную компиляцию и убрали GIL полностью."
Анализ:
{
    "analysis": "КРИТИЧЕСКАЯ ОШИБКА: Python 4.0 не существует. Кандидат выдумывает факты.",
    "is_correct": false,
    "correctness_score": 0,
    "confidence_level": "high",
    "red_flags": ["Галлюцинация про Python 4.0", "Выдуманные факты"],
    "is_hallucination": true,
    "hallucination_details": "Python 4.0 не существует. Последняя версия — Python 3.x. GIL не был полностью убран.",
    "is_question_from_candidate": false,
    "is_off_topic": false,
    "topic_detected": "версии Python",
    "difficulty_recommendation": "decrease",
    "recommendation": "Указать на ошибку и проверить базовые знания",
    "suggested_action": "challenge"
}

Пример 4 - Встречный вопрос:
Вопрос: "Расскажите про async/await"
Ответ кандидата: "А какие задачи вы решаете с помощью асинхронности в вашей компании?"
Анализ:
{
    "analysis": "Кандидат задал встречный вопрос вместо ответа. Это может быть попыткой уйти от темы или искренним интересом.",
    "is_correct": null,
    "correctness_score": 0,
    "confidence_level": "medium",
    "red_flags": [],
    "is_hallucination": false,
    "is_question_from_candidate": true,
    "is_off_topic": false,
    "topic_detected": "асинхронность",
    "difficulty_recommendation": "maintain",
    "recommendation": "Ответить на вопрос кандидата и вернуться к теме async/await",
    "suggested_action": "answer_question"
}

Пример 5 - Edge case "не знаю":
Вопрос: "Что такое метакласс?"
Ответ кандидата: "Честно говоря, не знаю. Не сталкивался с этим."
Анализ:
{
    "analysis": "Кандидат честно признал незнание темы. Это лучше, чем выдумывать.",
    "is_correct": false,
    "correctness_score": 0,
    "confidence_level": "low",
    "red_flags": ["Не знает метаклассы"],
    "is_hallucination": false,
    "is_question_from_candidate": false,
    "is_off_topic": false,
    "is_dont_know": true,
    "topic_detected": "метаклассы",
    "difficulty_recommendation": "decrease",
    "recommendation": "Похвалить за честность, кратко объяснить и перейти к более простой теме",
    "suggested_action": "simplify"
}
"""
    
    def analyze(self, user_msg: str, history: List[Dict], 
                current_difficulty: int, topics_covered: List[str] = None) -> Dict[str, Any]:
        """
        Анализирует ответ кандидата с использованием few-shot примеров.
        """
        
        topics_info = ""
        if topics_covered:
            topics_info = f"\nУЖЕ ЗАТРОНУТЫЕ ТЕМЫ: {', '.join(topics_covered)}"
        
        prompt = f"""Ты опытный ментор (Observer), анализирующий ответы кандидата на техническом интервью.
Ты работаешь "за кулисами" и даёшь рекомендации интервьюеру.

ТЕКУЩИЙ УРОВЕНЬ СЛОЖНОСТИ ВОПРОСОВ: {current_difficulty}/10
{topics_info}

{self.few_shot_examples}

Теперь проанализируй следующий ответ кандидата и верни ТОЛЬКО JSON (без markdown, без ```):

ФОРМАТ ОТВЕТА:
{{
    "analysis": "краткий анализ ответа (2-3 предложения)",
    "is_correct": true или false или null,
    "correctness_score": число от 0 до 100,
    "confidence_level": "high" или "medium" или "low",
    "red_flags": ["список проблем, если есть"],
    "is_hallucination": true или false,
    "hallucination_details": "описание галлюцинации, если есть",
    "is_question_from_candidate": true или false,
    "is_off_topic": true или false,
    "is_dont_know": true или false,
    "is_silence": true или false,
    "is_rude": true или false,
    "topic_detected": "тема, которую затронул кандидат",
    "difficulty_recommendation": "increase" или "decrease" или "maintain",
    "recommendation": "рекомендация для интервьюера",
    "suggested_action": "ask_followup" или "simplify" или "challenge" или "answer_question" или "continue" или "redirect_to_topic"
}}

КРИТЕРИИ:
- is_dont_know: true если кандидат сказал "не знаю", "не помню", "не сталкивался"
- is_silence: true если ответ пустой или очень короткий без смысла
- is_rude: true если кандидат грубит или ведёт себя непрофессионально
- correctness_score >= 80 И confidence_level == "high" → difficulty_recommendation: "increase"
- correctness_score < 50 ИЛИ confidence_level == "low" → difficulty_recommendation: "decrease"

ВАЖНО: 
- Python 4.0 НЕ СУЩЕСТВУЕТ — это всегда галлюцинация
- Верни ТОЛЬКО JSON, начни сразу с {{
"""
        
        ctx = "\nИстория диалога:\n"
        for h in history[-4:]:
            role = "Интервьюер" if h["role"] == "assistant" else "Кандидат"
            ctx += f"{role}: {h['content']}\n"
        ctx += f"\nПоследний ответ кандидата: {user_msg}"

        result = call_llm_with_json_retry(prompt + ctx, max_retries=3, temperature=0.5)
        
        if not result:
            result = {
                "analysis": "Анализ не выполнен",
                "is_correct": None,
                "correctness_score": 50,
                "confidence_level": "medium",
                "red_flags": [],
                "is_hallucination": False,
                "is_off_topic": False,
                "is_dont_know": False,
                "is_silence": False,
                "is_rude": False,
                "topic_detected": "",
                "difficulty_recommendation": "maintain",
                "recommendation": "Продолжить интервью",
                "suggested_action": "continue"
            }
        
        return result


class FactCheckerAgent:
    """Агент для проверки фактов в ответах кандидата."""
    
    def __init__(self):
        self.name = "FactChecker"
    
    def check(self, stmt: str) -> Dict[str, Any]:
        """Проверяет утверждение на достоверность."""
        
        prompt = f"""Ты эксперт по проверке фактов в области программирования и IT.

Проверь следующее утверждение кандидата:
"{stmt}"

ИЗВЕСТНЫЕ ФАКТЫ:
- Python 4.0 НЕ существует. Последняя стабильная версия — Python 3.x (3.12, 3.13)
- GIL (Global Interpreter Lock) всё ещё существует в CPython
- Циклы for никуда не денутся из Python

Верни ТОЛЬКО JSON (без markdown):
{{
    "is_true": true или false,
    "explanation": "подробное объяснение",
    "correct_info": "правильная информация (если утверждение ложно)"
}}
"""
        
        result = call_llm_with_json_retry(prompt, max_retries=3, temperature=0.3)
        
        if not result:
            result = {
                "is_true": False,
                "explanation": "Не удалось проверить утверждение",
                "correct_info": ""
            }
        
        return result


class FeedbackGeneratorAgent:
    """Агент для генерации итогового отчёта по интервью."""
    
    def __init__(self):
        self.name = "FeedbackGenerator"
    
    def generate(self, history: List[Dict], thoughts: List[InternalThought],
                 candidate: Dict[str, str], difficulty_history: List[int],
                 questions_asked: List[Dict] = None) -> Dict[str, Any]:
        """Генерирует структурированный отчёт по интервью."""
        
        hist_str = ""
        for h in history:
            role = "Интервьюер" if h["role"] == "assistant" else "Кандидат"
            hist_str += f"{role}: {h['content']}\n"
        
        thoughts_str = ""
        for t in thoughts:
            thoughts_str += f"[{t.from_agent} -> {t.to_agent}]: {t.content}\n"

        questions_info = ""
        if questions_asked:
            failed_questions = [q for q in questions_asked if q.get("is_correct") == False or q.get("correctness_score", 100) < 50]
            if failed_questions:
                questions_info = "\n\nВОПРОСЫ, НА КОТОРЫЕ КАНДИДАТ ОТВЕТИЛ НЕПРАВИЛЬНО:\n"
                for q in failed_questions:
                    questions_info += f"- Вопрос: {q.get('question', 'N/A')}\n  Ответ кандидата: {q.get('answer', 'N/A')}\n  Оценка: {q.get('correctness_score', 0)}%\n\n"
        
        avg_difficulty = sum(difficulty_history) / len(difficulty_history) if difficulty_history else 0
        max_difficulty = max(difficulty_history) if difficulty_history else 0
        
        prompt = f"""Ты эксперт по оценке кандидатов на технических собеседованиях.

На основе истории интервью сгенерируй подробный структурированный отчёт.

ИСТОРИЯ ДИАЛОГА:
{hist_str}

ВНУТРЕННИЕ МЫСЛИ АГЕНТОВ:
{thoughts_str}
{questions_info}

ИНФОРМАЦИЯ О КАНДИДАТЕ:
- Имя: {candidate.get('name', 'Unknown')}
- Позиция: {candidate.get('position', 'Unknown')}
- Заявленный грейд: {candidate.get('grade', 'Unknown')}
- Опыт: {candidate.get('experience', 'Unknown')}

СТАТИСТИКА СЛОЖНОСТИ:
- Средний уровень: {avg_difficulty:.1f}/10
- Максимальный уровень: {max_difficulty}/10
- История: {difficulty_history}

ВАЖНЫЕ ИНСТРУКЦИИ:
1. В knowledge_gaps для КАЖДОГО неправильного ответа укажи ПОДРОБНЫЙ правильный ответ
2. В resources укажи КОНКРЕТНЫЕ URL (docs.python.org, realpython.com, habr.com)
3. Оценивай реальный уровень кандидата, а не заявленный

Верни ТОЛЬКО JSON (без markdown, без ```):
{{
    "verdict": {{
        "grade": "Junior/Middle/Senior (реальный уровень)",
        "hiring_recommendation": "Strong Hire/Hire/No Hire",
        "confidence_score": число от 0 до 100
    }},
    "technical_review": {{
        "confirmed_skills": ["навык 1 с пояснением", "навык 2 с пояснением"],
        "knowledge_gaps": [
            {{
                "topic": "тема",
                "question": "вопрос",
                "candidate_answer": "что ответил кандидат",
                "correct_answer": "ПОДРОБНЫЙ правильный ответ (3-5 предложений)"
            }}
        ]
    }},
    "soft_skills": {{
        "clarity": число от 1 до 10,
        "honesty": число от 1 до 10,
        "engagement": число от 1 до 10,
        "comments": "комментарии"
    }},
    "roadmap": {{
        "topics_to_improve": ["тема 1", "тема 2"],
        "resources": [
            "https://docs.python.org/3/tutorial/ - Официальный туториал Python",
            "https://realpython.com/python-basics/ - Основы Python",
            "конкретные ссылки по темам"
        ]
    }},
    "difficulty_analysis": {{
        "average_level": {avg_difficulty:.1f},
        "max_reached": {max_difficulty},
        "progression": "описание как менялась сложность"
    }},
    "summary": "общее резюме (3-5 предложений)"
}}
"""

        result = call_llm_with_json_retry(prompt, max_retries=3, temperature=0.5)
        
        if not result:
            result = {
                "verdict": {"grade": "Unknown", "hiring_recommendation": "No Hire", "confidence_score": 0},
                "technical_review": {"confirmed_skills": [], "knowledge_gaps": []},
                "soft_skills": {"clarity": 5, "honesty": 5, "engagement": 5, "comments": ""},
                "roadmap": {
                    "topics_to_improve": [], 
                    "resources": [
                        "https://docs.python.org/3/tutorial/ - Официальный туториал Python",
                        "https://realpython.com/ - Практические руководства по Python"
                    ]
                },
                "difficulty_analysis": {"average_level": avg_difficulty, "max_reached": max_difficulty, "progression": ""},
                "summary": "Не удалось сгенерировать отчет"
            }

        if not result.get("roadmap", {}).get("resources"):
            result["roadmap"]["resources"] = [
                "https://docs.python.org/3/tutorial/ - Официальный туториал Python",
                "https://realpython.com/ - Практические руководства по Python",
                "https://habr.com/ru/hubs/python/ - Статьи о Python на Хабре"
            ]
        
        return result


class MultiAgentInterviewCoach:
    """
    Главный класс-координатор мультиагентной системы.
    
    Оркестрирует взаимодействие между агентами и управляет
    состоянием интервью.
    """
    
    def __init__(self, team_name: str = "Multi-Agent Interview Coach"):
        self.team_name = team_name
        self.interviewer = InterviewerAgent()
        self.mentor = MentorAgent()
        self.fact_checker = FactCheckerAgent()
        self.feedback_gen = FeedbackGeneratorAgent()
        
        self.candidate: Dict[str, str] = {}
        self.history: List[Dict] = []
        self.turns: List[Dict] = []
        self.thoughts: List[InternalThought] = []
        self.red_flags: List[str] = []
        self.turn_num = 0
        self.finished = False
        
        self.difficulty = 2
        self.correct_streak = 0
        self.difficulty_history: List[int] = []
        self.topics_covered: List[str] = []
        self.questions_asked: List[Dict] = []
    
    def start(self, name: str, position: str, grade: str, exp: str) -> str:
        """Инициализирует интервью и возвращает первое сообщение интервьюера."""
        
        self.candidate = {
            "name": name,
            "position": position,
            "grade": grade,
            "experience": exp
        }

        grade_lower = grade.lower()
        if "senior" in grade_lower:
            self.difficulty = 8
        elif "middle" in grade_lower:
            self.difficulty = 6
        elif "junior+" in grade_lower or "junior plus" in grade_lower:
            self.difficulty = 5
        elif "junior" in grade_lower:
            self.difficulty = 3
        else:
            self.difficulty = 2
        
        self.difficulty_history.append(self.difficulty)
        
        resp = self.interviewer.generate_response(
            self.candidate, [], "", 0, 
            self.difficulty, self.correct_streak,
            self.topics_covered
        )
        self.history.append({"role": "assistant", "content": resp})
        
        return resp

    def _adjust_difficulty(self, analysis: Dict[str, Any]):
        recommendation = analysis.get("difficulty_recommendation", "maintain")
        correctness = analysis.get("correctness_score", 50)
        is_correct = analysis.get("is_correct", False)

        if is_correct and correctness >= 80:
            self.correct_streak += 1
        else:
            self.correct_streak = 0

        if is_correct and (recommendation == "increase" or self.correct_streak >= 2):
            self.difficulty = min(10, self.difficulty + 1)
            self.correct_streak = 0
        elif recommendation == "decrease" or (not is_correct and correctness < 40):
            self.difficulty = max(1, self.difficulty - 1)

        self.difficulty_history.append(self.difficulty)

    
    def _detect_edge_case(self, user_msg: str, analysis: Dict[str, Any]) -> Optional[str]:
        """
        Определяет edge case на основе ответа пользователя.
        """
        msg_lower = user_msg.lower().strip()

        if len(msg_lower) < 3 or msg_lower in [".", "...", "-", "—"]:
            return "silence"

        dont_know_phrases = [
            "не знаю", "не помню", "не сталкивался", "не уверен",
            "затрудняюсь", "не могу ответить", "пропустить", "skip",
            "не изучал", "не работал с этим", "не приходилось"
        ]
        if any(phrase in msg_lower for phrase in dont_know_phrases) or analysis.get("is_dont_know"):
            return "dont_know"

        rude_phrases = [
            "отстань", "достал", "надоел", "глупый вопрос", "тупой",
            "идиот", "дурак", "пошёл", "отвали", "заткнись"
        ]
        if any(phrase in msg_lower for phrase in rude_phrases) or analysis.get("is_rude"):
            return "rude"

        troll_indicators = [
            "42", "потому что", "а зачем", "не скажу", "угадай",
            "секрет", "магия", "потому что гладиолус"
        ]
        if msg_lower in troll_indicators or (len(msg_lower) < 10 and "?" in msg_lower):
            if analysis.get("correctness_score", 50) < 10 and not analysis.get("is_question_from_candidate"):
                return "troll"
        
        return None
    
    def process(self, user_msg: str) -> str:
        """
        Обрабатывает ответ кандидата.
        
        Выполняет анализ через MentorAgent (Observer), проверку фактов через
        FactCheckerAgent при необходимости, и генерирует ответ
        интервьюера.
        """
        
        self.turn_num += 1
        self.history.append({"role": "user", "content": user_msg})
        
        if any(cmd in user_msg.lower() for cmd in ["стоп", "stop", "фидбэк", "feedback"]):
            return self._gen_feedback()

        analysis = self.mentor.analyze(user_msg, self.history, self.difficulty, self.topics_covered)

        edge_case = self._detect_edge_case(user_msg, analysis)

        if analysis.get("topic_detected"):
            topic = analysis["topic_detected"]
            if topic and topic not in self.topics_covered:
                self.topics_covered.append(topic)

        if len(self.history) >= 2:
            last_question = self.history[-2]["content"] if self.history[-2]["role"] == "assistant" else ""
            self.questions_asked.append({
                "question": last_question,
                "answer": user_msg,
                "is_correct": analysis.get("is_correct"),
                "correctness_score": analysis.get("correctness_score", 0),
                "edge_case": edge_case
            })

        if edge_case not in ["silence", "dont_know", "rude", "troll"]:
            self._adjust_difficulty(analysis)
        elif edge_case in ["dont_know", "silence"]:
            self.difficulty = max(1, self.difficulty - 1)
            self.difficulty_history.append(self.difficulty)

        mentor_thought = InternalThought(
            from_agent="Observer",
            to_agent="Interviewer",
            content=f"Анализ: {analysis.get('analysis', '')}. "
                    f"Корректность: {analysis.get('correctness_score', '?')}%. "
                    f"Уверенность: {analysis.get('confidence_level', 'unknown')}. "
                    f"Сложность: {analysis.get('difficulty_recommendation', 'maintain')}. "
                    f"Edge case: {edge_case or 'none'}. "
                    f"Рекомендация: {analysis.get('recommendation', '')}"
        )
        self.thoughts.append(mentor_thought)

        fc_thought = None
        if analysis.get("is_hallucination"):
            fc_result = self.fact_checker.check(user_msg)
            
            if not fc_result.get("is_true", True):
                flag = f"ГАЛЛЮЦИНАЦИЯ: {fc_result.get('explanation', 'Некорректное утверждение')}. Правильная информация: {fc_result.get('correct_info', '')}"
                self.red_flags.append(flag)
                
                fc_thought = InternalThought(
                    from_agent="FactChecker",
                    to_agent="Interviewer",
                    content=f"ALERT: Пользователь галлюцинирует. {fc_result.get('explanation', '')}. "
                            f"Правильная информация: {fc_result.get('correct_info', '')}. "
                            f"Это критическая ошибка знаний. Пометь как 'red flag'."
                )
                self.thoughts.append(fc_thought)

        advice = analysis.get("recommendation", "")
        if fc_thought:
            advice = f"ВАЖНО: {fc_thought.content}"
        elif analysis.get("is_question_from_candidate"):
            advice = "Кандидат задал вопрос. ОБЯЗАТЕЛЬНО ответь на него кратко и профессионально, затем продолжи интервью."
        elif analysis.get("is_off_topic"):
            advice = "Кандидат ушёл от темы. Вежливо верни беседу к техническому интервью."
        elif edge_case == "dont_know":
            advice = "Кандидат признал незнание. Похвали за честность, КРАТКО объясни правильный ответ и перейди к более простой теме."
        elif edge_case == "silence":
            advice = "Кандидат молчит или дал пустой ответ. Подбодри его и переформулируй вопрос проще."
        elif edge_case == "rude":
            advice = "Кандидат ведёт себя непрофессионально. Сохраняй спокойствие, напомни о формате интервью."
        elif edge_case == "troll":
            advice = "Кандидат даёт несерьёзные ответы. Игнорируй и задай следующий вопрос."
        elif analysis.get("difficulty_recommendation") == "increase":
            advice += " Кандидат справляется хорошо — задай более сложный вопрос."
        elif analysis.get("difficulty_recommendation") == "decrease":
            advice += " Кандидат затрудняется — упрости следующий вопрос или дай подсказку."

        resp = self.interviewer.generate_response(
            self.candidate, self.history, advice, self.turn_num,
            self.difficulty, self.correct_streak,
            self.topics_covered, edge_case
        )
        
        self.history.append({"role": "assistant", "content": resp})

        thoughts_str = f"[Observer]: {mentor_thought.content}"
        if fc_thought:
            thoughts_str += f" [FactChecker]: {fc_thought.content}"
        
        last_msg = self.history[-3]["content"] if len(self.history) >= 3 else ""
        self.turns.append({
            "turn_id": self.turn_num,
            "agent_visible_message": last_msg,
            "user_message": user_msg,
            "internal_thoughts": thoughts_str,
            "difficulty_level": self.difficulty,
            "edge_case": edge_case,
            "timestamp": datetime.now().isoformat()
        })
        
        return resp
    
    def _gen_feedback(self) -> str:
        """Генерирует итоговый отчёт по интервью."""
        
        self.finished = True
        
        feedback = self.feedback_gen.generate(
            self.history, self.thoughts, self.candidate, 
            self.difficulty_history, self.questions_asked
        )
        
        report = f"""
# Итоговый отчёт по интервью

## Информация о кандидате

| Параметр | Значение |
|----------|----------|
| **Имя** | {self.candidate.get('name', 'N/A')} |
| **Позиция** | {self.candidate.get('position', 'N/A')} |
| **Заявленный грейд** | {self.candidate.get('grade', 'N/A')} |
| **Опыт** | {self.candidate.get('experience', 'N/A')} |

---

## Вердикт

| Параметр | Значение |
|----------|----------|
| **Реальный грейд** | {feedback['verdict']['grade']} |
| **Рекомендация** | {feedback['verdict']['hiring_recommendation']} |
| **Уверенность** | {feedback['verdict']['confidence_score']}% |

---

## Анализ сложности вопросов

| Метрика | Значение |
|---------|----------|
| **Начальный уровень** | {self.difficulty_history[0] if self.difficulty_history else 'N/A'}/10 |
| **Финальный уровень** | {self.difficulty_history[-1] if self.difficulty_history else 'N/A'}/10 |
| **Максимальный уровень** | {max(self.difficulty_history) if self.difficulty_history else 'N/A'}/10 |
| **Средний уровень** | {sum(self.difficulty_history)/len(self.difficulty_history):.1f}/10 |

**История изменения:** {' -> '.join(map(str, self.difficulty_history))}

---

## Технический обзор

### Подтвержденные навыки:
"""
        for skill in feedback['technical_review'].get('confirmed_skills', []):
            report += f"- {skill}\n"
        
        report += "\n### Пробелы в знаниях:\n"
        for gap in feedback['technical_review'].get('knowledge_gaps', []):
            if isinstance(gap, dict):
                report += f"- **{gap.get('topic', 'N/A')}**\n"
                if gap.get('question'):
                    report += f"  - Вопрос: {gap['question']}\n"
                if gap.get('candidate_answer'):
                    report += f"  - Ответ кандидата: {gap['candidate_answer']}\n"
                if gap.get('correct_answer'):
                    report += f"  - ✓ Правильный ответ: {gap['correct_answer']}\n"
            else:
                report += f"- {gap}\n"
        
        soft = feedback.get('soft_skills', {})
        report += f"""
---

## Soft Skills

| Навык | Оценка |
|-------|--------|
| Ясность изложения (Clarity) | {soft.get('clarity', 'N/A')}/10 |
| Честность (Honesty) | {soft.get('honesty', 'N/A')}/10 |
| Вовлеченность (Engagement) | {soft.get('engagement', 'N/A')}/10 |

**Комментарии:** {soft.get('comments', '')}

---

## Рекомендации по развитию (Roadmap)

### Темы для изучения:
"""
        for topic in feedback['roadmap'].get('topics_to_improve', []):
            report += f"- {topic}\n"
        
        report += "\n### Рекомендуемые ресурсы:\n"
        for res in feedback['roadmap'].get('resources', []):
            report += f"- {res}\n"
        
        report += f"""
---

## Резюме

{feedback.get('summary', '')}
"""
        
        self.final_feedback = feedback
        
        return report
    
    def get_log(self) -> Dict:
        """
        Возвращает полный лог интервью в формате, соответствующем ТЗ.
        """
        return {
            "participant_name": self.candidate.get("name", "Unknown"),
            "team_name": self.team_name,
            "timestamp": datetime.now().isoformat(),
            "candidate": self.candidate,
            "turns": self.turns,
            "topics_covered": self.topics_covered,
            "difficulty_history": self.difficulty_history,
            "red_flags": self.red_flags,
            "final_feedback": getattr(self, 'final_feedback', None)
        }
    
    def save_log(self, path: str = None) -> str:
        """Сохраняет лог интервью в JSON-файл."""
        
        if path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"logs/interview_log_{ts}.json"
            fixed_path = f"logs/fixed_format_log_{ts}.json"
        else:
            fixed_path = path.replace(".json", "_fixed.json")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(fixed_path, 'w', encoding='utf-8') as f:
            f.write(LogAdapter.to_json(self))
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.get_log(), f, ensure_ascii=False, indent=2)
        
        return path


def interactive_mode():
    """Интерактивный режим — пользователь отвечает на вопросы в терминале."""
    
    print("=" * 70)
    print("MULTI-AGENT INTERVIEW COACH")
    print("Интерактивный режим технического интервью")
    print("=" * 70)
    print()
    
    print("Введите данные кандидата:")
    name = input("  Имя: ").strip() or "Кандидат"
    position = input("  Позиция (например, Backend Developer): ").strip() or "Backend Developer"
    grade = input("  Грейд (Junior/Middle/Senior): ").strip() or "Junior"
    experience = input("  Опыт (кратко): ").strip() or "Начинающий разработчик"
    
    print()
    print("-" * 70)
    print(f"Кандидат: {name}")
    print(f"Позиция: {position}")
    print(f"Грейд: {grade}")
    print(f"Опыт: {experience}")
    print("-" * 70)
    
    coach = MultiAgentInterviewCoach()
    
    first_msg = coach.start(name, position, grade, experience)
    print(f"\n[Уровень сложности: {coach.difficulty}/10]")
    print(f"\nИнтервьюер: {first_msg}")
    
    print("\n" + "=" * 70)
    print("Команды: 'стоп' или 'фидбэк' — завершить интервью и получить отчёт")
    print("         'выход' — выйти без сохранения")
    print("=" * 70)
    
    while True:
        print()
        user_input = input("Вы: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() in ['выход', 'exit', 'quit', 'q']:
            print("\nВыход без сохранения...")
            break
        
        response = coach.process(user_input)
        print(f"\n[Уровень сложности: {coach.difficulty}/10]")
        print(f"\nИнтервьюер: {response}")
        
        if coach.finished:
            log_path = coach.save_log()
            print(f"\n{'=' * 70}")
            print(f"Лог интервью сохранён: {log_path}")
            print(f"{'=' * 70}")
            break
    
    return coach


def demo_mode():
    """Демо-режим с готовым сценарием."""
    
    print("=" * 70)
    print("MULTI-AGENT INTERVIEW COACH v2.0")
    print("Демо-режим (готовый сценарий)")
    print("=" * 70)
    
    coach = MultiAgentInterviewCoach()
    
    print("\n[Инициализация интервью]")
    print("-" * 70)
    print("Кандидат: Алекс")
    print("Позиция: Backend Developer")
    print("Грейд: Junior")
    print("Опыт: Пет-проекты на Django, немного SQL")
    print("-" * 70)
    
    first_msg = coach.start("Алекс", "Backend Developer", "Junior", "Пет-проекты на Django, немного SQL")
    print(f"\n[Уровень сложности: {coach.difficulty}/10]")
    print(f"\nИнтервьюер: {first_msg}")
    print("-" * 70)
    
    scenario = [
        ("Ход 1 (Приветствие)", 
         "Привет. Я Алекс, претендую на позицию Junior Backend Developer. Знаю Python, SQL и Git."),
        
        ("Ход 2 (Хороший ответ)", 
         "Список в Python - это изменяемая упорядоченная коллекция элементов. Основные методы: append() для добавления элемента в конец, pop() для удаления последнего элемента, insert() для вставки по индексу. Списки поддерживают срезы и итерацию. Словарь - это неупорядоченная коллекция пар ключ-значение, где ключи должны быть хешируемыми."),
        
        ("Ход 3 (Галлюцинация)", 
         "Честно говоря, я читал на Хабре, что в Python 4.0 циклы for уберут и заменят на нейронные связи, поэтому я их не учу."),
        
        ("Ход 4 (Встречный вопрос)", 
         "Слушайте, а какие задачи вообще будут на испытательном сроке? Вы используете микросервисы?"),
        
        ("Ход 5 (Завершение)", 
         "Стоп игра. Давай фидбэк.")
    ]
    
    for step, response in scenario:
        print(f"\n{step}")
        print(f"Кандидат: {response}")
        
        agent_resp = coach.process(response)
        print(f"\n[Уровень сложности: {coach.difficulty}/10]")
        print(f"\nИнтервьюер: {agent_resp}")
        print("-" * 70)
    
    log_path = coach.save_log()
    print(f"\n{'=' * 70}")
    print(f"Лог интервью сохранён: {log_path}")
    print(f"{'=' * 70}")
    
    return coach


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--demo":
            demo_mode()
        elif sys.argv[1] == "--interactive":
            interactive_mode()
        else:
            print("Использование:")
            print("  python interview_system_v2.py --interactive  # Интерактивный режим")
            print("  python interview_system_v2.py --demo         # Демо-режим")
    else:
        print("Использование:")
        print("  python interview_system_v2.py --interactive  # Интерактивный режим")
        print("  python interview_system_v2.py --demo         # Демо-режим")
