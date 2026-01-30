"""
Модуль адаптера для логов.
Преобразует состояние интервью в фиксированный формат JSON.
"""
import json
from typing import Dict, Any, Union


class LogAdapter:
    """Адаптер для преобразования данных интервью в требуемый формат логов."""

    @staticmethod
    def _format_feedback_markdown(feedback: Dict[str, Any]) -> str:
        """Преобразует словарь фидбэка в Markdown-строку."""
        if not feedback:
            return "Интервью не завершено или отзыв отсутствует."

        md = []

        verdict = feedback.get("verdict", {})
        if verdict:
            md.append("### 1. Вердикт")
            md.append(f"- **Оценка (Grade):** {verdict.get('grade', 'N/A')}")
            md.append(f"- **Рекомендация:** {verdict.get('hiring_recommendation', 'N/A')}")
            md.append(f"- **Уверенность системы:** {verdict.get('confidence_score', 'N/A')}%")
            md.append("")

        tech = feedback.get("technical_review", {})
        if tech:
            md.append("### 2. Технический анализ")

            confirmed = tech.get("confirmed_skills", [])
            if confirmed:
                md.append("#### Подтвержденные навыки:")
                for skill in confirmed:
                    md.append(f"- {skill}")
                md.append("")

            gaps = tech.get("knowledge_gaps", [])
            if gaps:
                md.append("#### Пробелы в знаниях:")
                for gap in gaps:
                    topic = gap.get('topic', 'N/A')
                    md.append(f"- **{topic}**")
                    md.append(f"  - *Вопрос:* {gap.get('question', 'N/A')}")
                    md.append(f"  - *Ответ кандидата:* {gap.get('candidate_answer', 'N/A')}")
                md.append("")

        soft = feedback.get("soft_skills", {})
        if soft:
            md.append("### 3. Soft Skills")
            md.append(f"- **Ясность изложения:** {soft.get('clarity', 'N/A')}/5")
            md.append(f"- **Честность:** {soft.get('honesty', 'N/A')}/5")
            md.append(f"- **Вовлеченность:** {soft.get('engagement', 'N/A')}/5")
            if soft.get('comments'):
                md.append(f"- **Комментарий:** {soft.get('comments')}")
            md.append("")

        roadmap = feedback.get("roadmap", {})
        if roadmap:
            md.append("### 4. План развития")
            topics = roadmap.get("topics_to_improve", [])
            if topics:
                md.append("#### Что подтянуть:")
                for t in topics:
                    md.append(f"- {t}")

            resources = roadmap.get("resources", [])
            if resources:
                md.append("#### Рекомендуемые ресурсы:")
                for r in resources:
                    md.append(f"- {r}")
            md.append("")

        summary = feedback.get("summary")
        if summary:
            md.append("### Итоговое резюме")
            md.append(summary)

        return "\n".join(md).strip()

    @staticmethod
    def transform(coach_or_state: Any) -> Dict[str, Any]:
        """
        Преобразует объект MultiAgentInterviewCoach или InterviewState в формат:
        {
          "participant_name": "...",
          "turns": [
            {
              "turn_id": 1,
              "agent_visible_message": "...",
              "user_message": "...",
              "internal_thoughts": "..."
            }
          ],
          "final_feedback": "..."
        }
        """
        is_dict = isinstance(coach_or_state, dict)

        if is_dict:
            participant_name = coach_or_state.get("candidate", {}).get("name", "Unknown")
            raw_turns = coach_or_state.get("turns", [])
            raw_feedback = coach_or_state.get("final_feedback", {})
        else:
            candidate = getattr(coach_or_state, "candidate", {})
            participant_name = candidate.get("name", "Unknown") if isinstance(candidate, dict) else "Unknown"
            raw_turns = getattr(coach_or_state, "turns", [])
            raw_feedback = getattr(coach_or_state, "final_feedback", {})

        turns = []
        for turn in raw_turns:
            turns.append({
                "turn_id": turn.get("turn_id"),
                "agent_visible_message": turn.get("agent_visible_message"),
                "user_message": turn.get("user_message"),
                "internal_thoughts": turn.get("internal_thoughts")
            })

        if isinstance(raw_feedback, dict) and raw_feedback:
            final_feedback_str = LogAdapter._format_feedback_markdown(raw_feedback)
        else:
            final_feedback_str = str(raw_feedback) if raw_feedback else "Интервью не завершено или отзыв отсутствует."

        return {
            "participant_name": participant_name,
            "turns": turns,
            "final_feedback": final_feedback_str
        }

    @staticmethod
    def to_json(coach_or_state: Any, indent: int = 2) -> str:
        """Возвращает преобразованный лог в формате JSON-строки."""
        data = LogAdapter.transform(coach_or_state)
        return json.dumps(data, ensure_ascii=False, indent=indent)


def save_fixed_log(coach_or_state: Any, file_path: str):
    """Сохраняет преобразованный лог в файл в фиксированном формате."""
    log_json = LogAdapter.to_json(coach_or_state)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(log_json)
    return file_path
