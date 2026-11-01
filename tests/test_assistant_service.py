import pytest

from medical_assistant.application.assistant_service import AssistantService
from medical_assistant.application.exceptions import LLMProviderException
from tests.conftest import FakeProvider, FailingProvider


def test_assistant_service_works() -> None:
    assistant_service = AssistantService(provider=FakeProvider())

    assert assistant_service.answer(question="How are you?") == "Testowa odpowiedź na: How are you?"


def test_assistant_service_fails() -> None:
    assistant_service = AssistantService(provider=FailingProvider())

    with pytest.raises(LLMProviderException):
        assistant_service.answer(question="How are you?")
