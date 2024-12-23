from typing import Iterable

import pytest

from sdk.python.goodfire.api.chat.client import ChatMessage
from sdk.python.goodfire.api.chat.interfaces import (
    ChatCompletion,
    LogitsResponse,
    StreamingChatCompletionChunk,
)
from sdk.python.goodfire.api.exceptions import RequestFailedException


def test_chat_logits_base_case(goodfire_client, variant_small, variant_medium):
    response = goodfire_client.chat.logits(
        messages=[ChatMessage(role="user", content="Hello how are you?")],
        model=variant_small,
    )
    assert isinstance(response, LogitsResponse)

    response = goodfire_client.chat.logits(
        messages=[ChatMessage(role="user", content="Hello how are you?")],
        model=variant_medium,
    )
    assert isinstance(response, LogitsResponse)


def test_chat_logits_raises_when_list_of_messages_is_out_of_bounds(
    goodfire_client, variant_small
):
    expected_message = "List should have at most 128 items after validation, not 129"
    with pytest.raises(RequestFailedException, match=expected_message):
        goodfire_client.chat.logits(
            messages=[
                ChatMessage(role="user", content="Hello how are you?")
                for _ in range(129)
            ],
            model=variant_small,
        )

    expected_message = "List should have at least 1 item after validation, not 0"
    with pytest.raises(RequestFailedException, match=expected_message):
        goodfire_client.chat.logits(
            messages=[],
            model=variant_small,
        )


def test_chat_logits_raises_when_unsupported_model_is_provided(goodfire_client):
    expected_message = "Input should be 'meta-llama/Meta-Llama-3.1-8B-Instruct' or 'meta-llama/Llama-3.3-70B-Instruct'"
    with pytest.raises(RequestFailedException, match=expected_message):
        goodfire_client.chat.logits(
            messages=[ChatMessage(role="user", content="Hello how are you?")],
            model="unsupported-model",
        )


def test_chat_completions_base_case(goodfire_client, variant_small, variant_medium):
    response = goodfire_client.chat.completions.create(
        messages=[ChatMessage(role="user", content="Hello how are you?")],
        model=variant_small,
        stream=False,
    )
    assert isinstance(response, ChatCompletion)

    response = goodfire_client.chat.completions.create(
        messages=[ChatMessage(role="user", content="Hello how are you?")],
        model=variant_medium,
        stream=False,
    )
    assert isinstance(response, ChatCompletion)

    response = goodfire_client.chat.completions.create(
        messages=[ChatMessage(role="user", content="Hello how are you?")],
        model=variant_small,
        stream=True,
    )
    assert isinstance(response, Iterable)

    sample_from_stream = None
    for token in goodfire_client.chat.completions.create(
        [{"role": "user", "content": "Hello. How are you?"}],
        model=variant_small,
        stream=True,
        max_completion_tokens=10,
    ):
        sample_from_stream = token

    assert isinstance(sample_from_stream, StreamingChatCompletionChunk)

    sample_from_stream = None
    for token in goodfire_client.chat.completions.create(
        [{"role": "user", "content": "Hello. How are you?"}],
        model=variant_medium,
        stream=True,
        max_completion_tokens=10,
    ):
        sample_from_stream = token

    assert isinstance(sample_from_stream, StreamingChatCompletionChunk)


def test_chat_completions_raises_when_unsupported_model_is_provided(goodfire_client):
    expected_message = "Input should be 'meta-llama/Meta-Llama-3.1-8B-Instruct' or 'meta-llama/Llama-3.3-70B-Instruct'"
    with pytest.raises(RequestFailedException, match=expected_message):
        goodfire_client.chat.completions.create(
            messages=[ChatMessage(role="user", content="Hello how are you?")],
            model="unsupported-model",
        )


def test_chat_completions_raises_when_list_of_messages_is_out_of_bounds(
    goodfire_client, variant_small
):
    expected_message = "List should have at most 128 items after validation, not 130"
    with pytest.raises(RequestFailedException, match=expected_message):
        goodfire_client.chat.completions.create(
            messages=[
                ChatMessage(role="user", content="Hello how are you?")
                for _ in range(129)
            ],
            model=variant_small,
        )


def test_chat_completions_raises_when_top_p_is_out_of_bounds(
    goodfire_client, variant_small
):
    expected_message = "Input should be less than or equal to 1"
    with pytest.raises(RequestFailedException, match=expected_message):
        goodfire_client.chat.completions.create(
            messages=[ChatMessage(role="user", content="Hello how are you?")],
            model=variant_small,
            top_p=1.1,
        )

    expected_message = "Input should be greater than or equal to 0"
    with pytest.raises(RequestFailedException, match=expected_message):
        goodfire_client.chat.completions.create(
            messages=[ChatMessage(role="user", content="Hello how are you?")],
            model=variant_small,
            top_p=-0.1,
        )


def test_chat_completions_raises_when_temperature_is_out_of_bounds(
    goodfire_client, variant_small
):
    expected_message = "Input should be greater than or equal to 0"
    with pytest.raises(RequestFailedException, match=expected_message):
        goodfire_client.chat.completions.create(
            messages=[ChatMessage(role="user", content="Hello how are you?")],
            model=variant_small,
            temperature=-0.1,
        )

    expected_message = "Input should be less than or equal to 2"
    with pytest.raises(RequestFailedException, match=expected_message):
        goodfire_client.chat.completions.create(
            messages=[ChatMessage(role="user", content="Hello how are you?")],
            model=variant_small,
            temperature=2.1,
        )
