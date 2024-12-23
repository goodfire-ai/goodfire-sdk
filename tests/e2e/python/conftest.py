# import goodfire
import pytest

from cajal.autointerp.frontier_models.claude import Claude
from sdk.python import goodfire
from sdk.tests.e2e.python.config import TestConfig

test_config = TestConfig()


@pytest.fixture
def goodfire_client():
    return goodfire.Client(
        api_key=test_config.E2E_GOODFIRE_API_KEY,
        base_url=test_config.E2E_GOODFIRE_SDK_CLIENT_BASE_URL,
    )


@pytest.fixture
def claude_client():
    return Claude(
        test_config.E2E_ANTHROPIC_API_KEY, model_name="claude-3-haiku-20240307"
    )


@pytest.fixture
def variant_small():
    return goodfire.Variant(
        base_model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    )


@pytest.fixture
def variant_medium():
    return goodfire.Variant(
        base_model="meta-llama/Llama-3.3-70B-Instruct",
    )
