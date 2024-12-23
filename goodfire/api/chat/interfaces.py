from typing import Literal, Optional

from pydantic import BaseModel
from typing_extensions import TypedDict


class StreamingDelta(BaseModel):
    role: str
    content: str


class StreamingChoice(BaseModel):
    index: int
    delta: StreamingDelta
    gf_token_index: int
    finish_reason: Optional[str]


class StreamingChatCompletionChunk(BaseModel):
    object: str
    id: str
    created: Optional[int]
    model: str
    system_fingerprint: str
    gf_event_names: Optional[list[str]] = None
    choices: list[StreamingChoice]


class ChatMessage(TypedDict):
    role: Literal["assistant", "system", "user"]
    content: str


class ChatResponse(BaseModel):
    role: Literal["assistant", "system", "user"]
    content: str


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatResponse
    finish_reason: Optional[str]


class ChatCompletion(BaseModel):
    id: str
    object: str
    created: Optional[int]
    model: str
    system_fingerprint: str
    gf_event_names: Optional[list[str]] = None
    choices: list[ChatCompletionChoice]


class LogitsResponse(BaseModel):
    logits: dict[str, float]
