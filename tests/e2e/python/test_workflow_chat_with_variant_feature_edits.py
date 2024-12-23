from typing import Literal

import pytest

from cajal.lib.prompt import LanguageModelPrompt

pytestmark = pytest.mark.asyncio


def _pirate_detection_prompt(sample: str):
    return LanguageModelPrompt(
        f"""A user submitted this content:

        <content>
        {sample}
        </content>

        Reply with "Yes" only if the content sounds like a pirate. Reply with "No" otherwise.

        At the end return your answer in <response> and </response> tags.
        """
    )


async def _is_response_in_pirate(response: str, claude_client) -> Literal["Yes", "No"]:
    claude_response = await claude_client.chat(
        [{"role": "user", "content": _pirate_detection_prompt(response)}],
        temperature=0.2,
        max_tokens_to_sample=500,
    )

    claude_response_content: str = claude_response[0]["content"]
    claude_answer = "No"

    try:
        claude_answer = claude_response_content.split("<response>")[1].split(
            "</response>"
        )[0]
    except IndexError:
        pass

    return claude_answer


async def test_chat_with_variant_feature_edits(
    goodfire_client, variant_small, claude_client
):
    response_content = ""

    for token in goodfire_client.chat.completions.create(
        [{"role": "user", "content": "Hello. How are you?"}],
        model=variant_small,
        stream=True,
        max_completion_tokens=50,
    ):
        response_content += token.choices[0].delta.content
    print(response_content)
    is_response_in_pirate = await _is_response_in_pirate(
        response_content, claude_client
    )

    assert is_response_in_pirate == "No"

    features = goodfire_client.features.search(
        "talk like a pirate", model=variant_small, top_k=5
    )
    print(features[0])
    variant_small.set(features[0], 0.8)

    response_content = ""
    for token in goodfire_client.chat.completions.create(
        [{"role": "user", "content": "Hello. How are you?"}],
        model=variant_small,
        stream=True,
        max_completion_tokens=50,
    ):
        response_content += token.choices[0].delta.content
    print(response_content)
    is_response_in_pirate = await _is_response_in_pirate(
        response_content, claude_client
    )

    assert is_response_in_pirate == "Yes"
