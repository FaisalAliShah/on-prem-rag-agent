from __future__ import annotations

import re
from typing import Any


def answer_contact_query(question: str, chunks: list[dict[str, Any]]) -> str | None:
    question_lower = question.lower()
    if not any(term in question_lower for term in ("email", "url", "link", "linkedin", "github", "phone", "contact")):
        return None

    text = "\n".join(chunk.get("text", "") for chunk in chunks)
    emails = sorted(set(re.findall(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", text)))
    text_without_emails = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", " ", text)
    urls = sorted(
        set(
            match.rstrip(".,);")
            for match in re.findall(
                r"(?:https?://)?(?:www\.)?[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+(?:/[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=%-]+)?",
                text_without_emails,
            )
        )
    )
    phones = sorted(set(re.findall(r"\+?\d[\d\s().-]{7,}\d", text)))

    requested_lines: list[str] = []
    if "email" in question_lower and emails:
        requested_lines.append("Email: " + ", ".join(emails))
    if any(term in question_lower for term in ("url", "link", "linkedin", "github")) and urls:
        requested_lines.append("URLs: " + ", ".join(urls))
    if any(term in question_lower for term in ("phone", "contact")) and phones:
        requested_lines.append("Phone: " + ", ".join(phones))

    if requested_lines:
        return "\n".join(requested_lines)
    return None
