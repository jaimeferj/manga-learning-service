from __future__ import annotations

import json
import logging
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

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


AiLearningAction = Literal["translate", "grammar", "vocabulary", "tone", "cards"]


class AiLearningContext(BaseModel):
    previous_line: str | None = None
    next_line: str | None = None
    speaker: str | None = None
    scene: str | None = None


class AiLearnRequest(BaseModel):
    sentence: str = Field(min_length=1)
    action: AiLearningAction
    context: AiLearningContext = Field(default_factory=AiLearningContext)
    level: str = "beginner-to-lower-intermediate"


class AiLearningSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    content: str


class AiLearningPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sections: list[AiLearningSection] = Field(min_length=1)


class AiLearnResponse(BaseModel):
    provider: str
    action: AiLearningAction
    sentence: str
    sections: list[AiLearningSection]


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

SYSTEM_LEARN = """You are a Japanese tutor specialized in helping Spanish-speaking learners read manga.
Explain in Spanish, use natural Japanese, and add furigana when helpful. Be practical and concise.
Distinguish literal from natural meaning, explain important omissions, and identify manga-like language.
Never invent context; state uncertainty when context is insufficient.
Return exactly one JSON object matching this schema:
{"sections":[{"label":"Short section title","content":"Section explanation"}]}
The top-level object may contain only `sections`. Every section may contain only the string fields
`label` and `content`. Never replace them with action-specific fields, named objects, `items`, or arrays
of another shape. Use the same schema for every action.
Keep every `content` under 500 characters and the complete response under 3000 characters.
Output the JSON object only, without commentary or markdown fences."""


ACTION_PROMPTS: dict[AiLearningAction, str] = {
    "translate": (
        "Translate naturally. Return sections for Traducción natural, Traducción literal, "
        "Omitido o implícito, and Tono."
    ),
    "grammar": (
        "Explain only grammar needed to understand the sentence. Return sections for Estructura, "
        "Puntos gramaticales, Formas verbales/adjetivales, Partículas, and Elementos omitidos. "
        "For each key pattern include its meaning, use here, and one simple example."
    ),
    "vocabulary": (
        "Explain important vocabulary, not particles. For each useful word include reading, Spanish "
        "meaning, nuance here, register, a short example, and similar-word differences only when useful."
    ),
    "tone": (
        "Explain speech nuance. Cover politeness, emotion, character voice, endings/contractions, "
        "speaker relationship, real-life versus manga usage, and a neutral version when useful."
    ),
    "cards": (
        "Create only useful Anki-style cards. Include sentence, vocabulary, grammar, and cloze cards "
        "only when warranted. Give each card its Type, Front, Back, and Tags."
    ),
}

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
    except json.JSONDecodeError as initial_exc:
        decoder = json.JSONDecoder()
        object_start = cleaned.find("{")
        try:
            data, _end = decoder.raw_decode(cleaned[object_start:])
        except (json.JSONDecodeError, ValueError):
            logger.warning("provider %s returned non-json: %s", provider, cleaned[:200])
            raise ValueError(
                f"provider {provider} returned non-json output: {initial_exc}"
            ) from initial_exc
    if not isinstance(data, dict):
        raise ValueError(f"provider {provider} returned non-object json")
    return data
