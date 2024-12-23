import os

from dotenv import load_dotenv

load_dotenv()


class TestConfig:
    E2E_GOODFIRE_API_KEY = os.getenv("E2E_GOODFIRE_API_KEY", "")
    E2E_GOODFIRE_SDK_CLIENT_BASE_URL = os.getenv("E2E_GOODFIRE_SDK_CLIENT_BASE_URL", "")
    E2E_ANTHROPIC_API_KEY = os.getenv("E2E_ANTHROPIC_API_KEY", "")
    E2E_CONCURRENT_REQUESTS = os.getenv("E2E_CONCURRENT_REQUESTS", 32)
