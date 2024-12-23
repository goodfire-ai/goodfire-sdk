import asyncio
import concurrent.futures

import pytest

from sdk.python.goodfire.api.exceptions import RateLimitException


def _chat_completion(goodfire_client, variant):
    goodfire_client.chat.completions.create(
        [{"role": "user", "content": "Hello. How are you?"}],
        model=variant,
        stream=False,
        max_completion_tokens=10,
    )


def _chat_logits(goodfire_client, variant):
    goodfire_client.chat.logits(
        messages=[{"role": "user", "content": "Hello. How are you?"}],
        model=variant,
    )


@pytest.mark.skip(
    reason="Need to think through a better way to reset rate limit after test run"
)
def test_org_rate_limit_chat_completion(goodfire_client, variant_small):
    with pytest.raises(RateLimitException):
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(_chat_completion, goodfire_client, variant_small)
                for _ in range(20)
            ]
            # Wait for all futures and re-raise any exceptions
            for future in concurrent.futures.as_completed(futures):
                future.result()  # This will raise the RateLimitException when it occurs


@pytest.mark.skip(
    reason="Need to think through a better way to reset rate limit after test run"
)
@pytest.mark.asyncio
async def test_org_rate_limit_chat_logits(goodfire_client, variant_small):
    await asyncio.sleep(120)  # Wait for rate limit to be reset
    with pytest.raises(RateLimitException):
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(_chat_logits, goodfire_client, variant_small)
                for _ in range(20)
            ]
            # Wait for all futures and re-raise any exceptions
            for future in concurrent.futures.as_completed(futures):
                future.result()  # This will raise the RateLimitException when it occurs
