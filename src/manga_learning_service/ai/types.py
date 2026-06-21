from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class AiExplainRequest(BaseModel):
    text: str
    language: str = "ja"
    level: str = "intermediate"
    include_vocab: bool = True
    include_grammar: bool = True


class AiExplainResponse(BaseModel):
    provider: str
    explanation: str
    vocab: list[dict[str, str]] = Field(default_factory=list)
    grammar: list[str] = Field(default_factory=list)


class AiCardFieldsRequest(BaseModel):
    text: str
    language: str = "ja"
    reading: str | None = None
    model: str | None = None


class AiCardFieldsResponse(BaseModel):
    provider: str
    fields: dict[str, str]
    notes: list[str] = Field(default_factory=list)


SYSTEM_EXPLAIN = (
    "You are a Japanese language tutor. Respond in compact JSON with keys: "
    "`explanation` (markdown), `vocab` (list of {term, reading, meaning}), "
    "`grammar` (list of short bullet strings). Do not wrap in code fences."
)

SYSTEM_CARD_FIELDS = (
    "You are an Anki card generator. Produce a JSON object with Anki field "
    "name keys: at minimum `Expression`, `Reading`, `Meaning`. Add extra "
    "fields only if explicitly requested by the caller. Reply ONLY with "
    "the JSON object, no markdown fences."
)


def parse_json_response(provider: str, raw: str) -> dict[str, Any]:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(
            line for line in cleaned.splitlines() if not line.startswith("```")
        ).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.warning("provider %s returned non-json: %s", provider, cleaned[:200])
        raise ValueError(f"provider {provider} returned non-json output: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"provider {provider} returned non-object json")
    return data
