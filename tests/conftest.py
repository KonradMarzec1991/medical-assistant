import pytest
from fastapi.testclient import TestClient

from medical_assistant.application.assistant_service import AssistantService
from medical_assistant.application.exceptions import LLMProviderException
from medical_assistant.main import app, get_assistant_service


class FakeProvider:
    def __init__(self):
        self.system_prompt = None
        self.user_prompt = None

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.system_prompt = system_prompt
        self.system_prompt = user_prompt

        return f"Testowa odpowiedź na: {user_prompt}"


class FailingProvider:
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        raise LLMProviderException("Provider failed")


@pytest.fixture
def assistant_service() -> AssistantService:
    return AssistantService(provider=FakeProvider())


@pytest.fixture
def client(assistant_service: AssistantService):
    app.dependency_overrides[get_assistant_service] = lambda: assistant_service
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.pop(get_assistant_service)


@pytest.fixture
def failing_client():
    service = AssistantService(provider=FailingProvider())
    app.dependency_overrides[get_assistant_service] = lambda: service
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.pop(get_assistant_service, None)
