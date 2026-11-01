import pytest
from fastapi.testclient import TestClient

from medical_assistant.application.assistant_service import AssistantService
from medical_assistant.main import app, get_assistant_service

class FakeProvider:
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return f"Testowa odpowiedź na: {user_prompt}"


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
