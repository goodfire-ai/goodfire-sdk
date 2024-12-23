from typing import (
    Any,
    AsyncIterator,
    Iterable,
    Iterator,
    Literal,
    Optional,
    TypedDict,
    Union,
    overload,
)

import httpx
from pydantic import ValidationError
from typing_extensions import NotRequired

from ...utils.asyncio import run_async_safely
from ...variants.variants import SUPPORTED_MODELS, InferenceContext, Variant
from ..constants import PRODUCTION_BASE_URL, SSE_DONE
from ..exceptions import ServerErrorException
from ..utils import AsyncHTTPWrapper, HTTPWrapper
from .interfaces import (
    ChatCompletion,
    ChatMessage,
    LogitsResponse,
    StreamingChatCompletionChunk,
)

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant who should follow the users requests. Be brief and to the point, but also be friendly and engaging."


class SteeringConfig(TypedDict):
    nonzero_strength_threshold: NotRequired[float]
    min_nudge_entropy: NotRequired[float]
    max_nudge_entropy: NotRequired[float]


class AsyncChatAPICompletions:
    """OpenAI compatible chat completions API."""

    def __init__(self, api_key: str, base_url: str = PRODUCTION_BASE_URL):
        self.api_key = api_key
        self.base_url = base_url

        self._http = AsyncHTTPWrapper()

    def _get_headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
        }

    @overload
    async def create(
        self,
        messages: Iterable[Union[ChatMessage, dict[str, str]]],
        model: Union[SUPPORTED_MODELS, Variant],
        *,
        stream: Literal[False] = False,
        max_completion_tokens: Optional[int] = None,
        top_p: float = 0.9,
        temperature: float = 0.6,
        stop: Optional[Union[str, list[str]]] = None,
        seed: Optional[int] = 42,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        steering_config: Optional[SteeringConfig] = None,
        _disable_moderation: bool = False,
    ) -> ChatCompletion: ...

    @overload
    async def create(
        self,
        messages: Iterable[Union[ChatMessage, dict[str, str]]],
        model: Union[SUPPORTED_MODELS, Variant],
        *,
        stream: Literal[True] = True,
        max_completion_tokens: Optional[int] = None,
        top_p: float = 0.9,
        temperature: float = 0.6,
        stop: Optional[Union[str, list[str]]] = None,
        seed: Optional[int] = 42,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        steering_config: Optional[SteeringConfig] = None,
        _disable_moderation: bool = False,
    ) -> AsyncIterator[StreamingChatCompletionChunk]: ...

    async def create(
        self,
        messages: Iterable[Union[ChatMessage, dict[str, str]]],
        model: Union[SUPPORTED_MODELS, Variant],
        stream: bool = False,
        max_completion_tokens: Optional[int] = 2048,
        top_p: float = 0.9,
        temperature: float = 0.6,
        stop: Optional[Union[str, list[str]]] = ["<|eot_id|>", "<|begin_of_text|>"],
        timeout: Optional[int] = 320,
        seed: Optional[int] = 42,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        steering_config: Optional[SteeringConfig] = None,
        _disable_moderation: bool = False,
    ) -> Union[ChatCompletion, AsyncIterator[StreamingChatCompletionChunk]]:
        """Create a chat completion."""
        url = f"{self.base_url}/api/inference/v1/chat/completions"

        headers = self._get_headers()

        messages = [*messages]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})

        payload: dict[str, Any] = {
            "messages": messages,
            "stream": stream,
            "max_completion_tokens": max_completion_tokens,
            "top_p": top_p,
            "temperature": temperature,
            "stop": stop,
            "seed": seed,
            "disable_moderation": _disable_moderation,
        }

        if isinstance(model, str):
            payload["model"] = model
        else:
            payload["model"] = model.base_model

            payload["controller"] = model.controller.json()

            if steering_config:
                payload["controller"].update(steering_config)

        if stream:

            async def _stream_response() -> AsyncIterator[StreamingChatCompletionChunk]:
                try:
                    message_so_far = ""
                    seen_indices: set[int] = set()
                    async for chunk in await self._http.stream(
                        "POST",
                        url,
                        headers={
                            **headers,
                            "Accept": "text/event-stream",
                            "Connection": "keep-alive",
                        },
                        json=payload,
                        timeout=timeout,
                    ):
                        chunk = chunk.decode("utf-8")

                        if chunk == SSE_DONE:
                            break

                        subchunks = chunk.split("data: ")

                        for subchunk in subchunks:
                            if not subchunk:
                                continue

                            if SSE_DONE.strip() in subchunk.strip():
                                break

                            chunk = StreamingChatCompletionChunk.model_validate_json(
                                subchunk.strip()
                            )

                            if chunk.gf_event_names and isinstance(model, Variant):
                                for event_name in chunk.gf_event_names:
                                    context = InferenceContext(
                                        prompt=messages,
                                        response_so_far=message_so_far,
                                    )
                                    if event_name in model._handlers:
                                        if handler := model.variant._handlers.get(
                                            event_name
                                        ):
                                            handler(context)

                            if chunk.choices[0].gf_token_index in seen_indices:
                                continue

                            seen_indices.add(chunk.choices[0].gf_token_index)

                            message_so_far += chunk.choices[0].delta.content

                            yield chunk
                except (httpx.RemoteProtocolError, ValidationError):
                    raise ServerErrorException()

            return _stream_response()
        else:
            response = await self._http.post(
                url,
                headers={
                    **headers,
                    "Accept": "application/json",
                },
                json=payload,
                timeout=timeout,
            )

            try:
                response = ChatCompletion.model_validate(response.json())

                if response.gf_event_names and isinstance(model, Variant):
                    context = InferenceContext(
                        prompt=messages,
                        response_so_far=response.choices[0].message["content"],
                    )
                    for event_name in response.gf_event_names:
                        if event_name in model._handlers:
                            if handler := model._handlers.get(event_name):
                                handler(context)

                return response
            except ValidationError:
                raise ServerErrorException("Server error")


class AsyncChatAPI:
    """OpenAI compatible chat API.

    Example:
        >>> async for token in client.chat.completions.create(
        ...     [
        ...         {"role": "user", "content": "hello"}
        ...     ],
        ...     model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        ...     stream=True,
        ...     max_completion_tokens=50,
        ... ):
        ...     print(token.choices[0].delta.content, end="")
    """

    def __init__(self, api_key: str, base_url: str = PRODUCTION_BASE_URL):
        self.api_key = api_key
        self.base_url = base_url
        self.completions = AsyncChatAPICompletions(api_key, base_url)
        self._http = AsyncHTTPWrapper()

    async def logits(
        self,
        messages: Iterable[Union[ChatMessage, dict[str, str]]],
        model: Union[SUPPORTED_MODELS, Variant],
        top_k: Optional[int] = 10,
        filter_vocabulary: Optional[list[str]] = None,
    ) -> LogitsResponse:
        """Compute logits for a chat completion."""
        payload: dict[str, Any] = {
            "messages": messages,
            "top_k": top_k,
            "filter_vocabulary": filter_vocabulary,
        }

        if isinstance(model, str):
            payload["model"] = model
        else:
            payload["model"] = model.base_model

            payload["controller"] = model.controller.json()

        response = await self._http.post(
            f"{self.base_url}/api/inference/v1/chat-attribution/compute-logits",
            headers={
                **self._get_headers(),
            },
            json=payload,
        )

        return LogitsResponse.model_validate(response.json())

    def _get_headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        }


class ChatAPICompletions:
    def __init__(self, api_key: str, base_url: str = PRODUCTION_BASE_URL):
        self.api_key = api_key
        self.base_url = base_url

        self._http = HTTPWrapper()

    def _get_headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
        }

    @overload
    def create(
        self,
        messages: Iterable[Union[ChatMessage, dict[str, str]]],
        model: Union[SUPPORTED_MODELS, Variant],
        *,
        stream: Literal[False] = False,
        max_completion_tokens: Optional[int] = None,
        top_p: float = 0.9,
        temperature: float = 0.6,
        stop: Optional[Union[str, list[str]]] = None,
        seed: Optional[int] = 42,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        steering_config: Optional[SteeringConfig] = None,
        _disable_moderation: bool = False,
    ) -> ChatCompletion: ...

    @overload
    def create(
        self,
        messages: Iterable[Union[ChatMessage, dict[str, str]]],
        model: Union[SUPPORTED_MODELS, Variant],
        *,
        stream: Literal[True] = True,
        max_completion_tokens: Optional[int] = None,
        top_p: float = 0.9,
        temperature: float = 0.6,
        stop: Optional[Union[str, list[str]]] = None,
        seed: Optional[int] = 42,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        steering_config: Optional[SteeringConfig] = None,
        _disable_moderation: bool = False,
    ) -> Iterable[StreamingChatCompletionChunk]: ...

    def create(
        self,
        messages: Iterable[Union[ChatMessage, dict[str, str]]],
        model: Union[SUPPORTED_MODELS, Variant],
        stream: bool = False,
        max_completion_tokens: Optional[int] = 2048,
        top_p: float = 0.9,
        temperature: float = 0.6,
        stop: Optional[Union[str, list[str]]] = ["<|eot_id|>", "<|begin_of_text|>"],
        timeout: Optional[int] = 320,
        seed: Optional[int] = 42,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        steering_config: Optional[SteeringConfig] = None,
        _disable_moderation: bool = False,
    ) -> Union[ChatCompletion, Iterable[StreamingChatCompletionChunk]]:
        """Create a chat completion."""
        url = f"{self.base_url}/api/inference/v1/chat/completions"

        headers = self._get_headers()

        messages = [*messages]
        if system_prompt and messages[0]["role"] != "system":
            messages.insert(0, {"role": "system", "content": system_prompt})

        payload: dict[str, Any] = {
            "messages": messages,
            "stream": stream,
            "max_completion_tokens": max_completion_tokens,
            "top_p": top_p,
            "temperature": temperature,
            "stop": stop,
            "seed": seed,
            "disable_moderation": _disable_moderation,
        }

        if isinstance(model, str):
            payload["model"] = model
        else:
            payload["model"] = model.base_model

            payload["controller"] = model.controller.json()

            if steering_config:
                payload["controller"].update(steering_config)

        if stream:

            def _stream_response() -> Iterator[StreamingChatCompletionChunk]:
                message_so_far = ""
                try:
                    seen_indices: set[int] = set()
                    for chunk in self._http.stream(
                        "POST",
                        url,
                        headers={
                            **headers,
                            "Accept": "text/event-stream",
                            "Connection": "keep-alive",
                        },
                        json=payload,
                        timeout=timeout,
                    ):
                        chunk = chunk.decode("utf-8")

                        if chunk == SSE_DONE:
                            break

                        subchunks = chunk.split("data: ")

                        for subchunk in subchunks:
                            if not subchunk:
                                continue

                            if SSE_DONE.strip() in subchunk.strip():
                                break

                            chunk = StreamingChatCompletionChunk.model_validate_json(
                                subchunk.strip()
                            )

                            if chunk.gf_event_names and isinstance(model, Variant):
                                for event_name in chunk.gf_event_names:
                                    context = InferenceContext(
                                        prompt=messages,
                                        response_so_far=message_so_far,
                                    )
                                    if event_name in model._handlers:
                                        if handler := model._handlers.get(event_name):
                                            handler(context)

                                            handler(context)
                            if chunk.choices[0].gf_token_index in seen_indices:
                                continue

                            seen_indices.add(chunk.choices[0].gf_token_index)

                            message_so_far += chunk.choices[0].delta.content

                            yield chunk
                except (httpx.RemoteProtocolError, ValidationError):
                    raise ServerErrorException()

            return _stream_response()
        else:
            response = self._http.post(
                url,
                headers={
                    **headers,
                    "Accept": "application/json",
                },
                json=payload,
                timeout=timeout,
            )

            try:
                response = ChatCompletion.model_validate(response.json())

                if response.gf_event_names and isinstance(model, Variant):
                    context = InferenceContext(
                        prompt=messages,
                        response_so_far=response.choices[0].message["content"],
                    )
                    for event_name in response.gf_event_names:
                        if event_name in model._handlers:
                            if handler := model._handlers.get(event_name):
                                handler(context)

                return response
            except ValidationError:
                raise ServerErrorException("Server error")


class ChatAPI:
    def __init__(self, api_key: str, base_url: str = PRODUCTION_BASE_URL):
        self.api_key = api_key
        self.base_url = base_url
        self.completions = ChatAPICompletions(api_key, base_url)
        self._async_chat_api = AsyncChatAPI(api_key, base_url)

    def logits(
        self,
        messages: Iterable[Union[ChatMessage, dict[str, str]]],
        model: Union[SUPPORTED_MODELS, Variant],
        top_k: Optional[int] = 10,
        filter_vocabulary: Optional[list[str]] = None,
    ) -> LogitsResponse:
        """Compute logits for a chat completion."""
        return run_async_safely(
            self._async_chat_api.logits(messages, model, top_k, filter_vocabulary)
        )
