import pytest

from medical_assistant.application.assistant_service import AssistantService


class FakeProvider:
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return f"Testowa odpowiedź na: {user_prompt}"


@pytest.fixture
def assistant_service() -> AssistantService:
    return AssistantService(provider=FakeProvider())
