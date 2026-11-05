import pytest

from medical_assistant.application.assistant_service import AssistantService
from medical_assistant.application.exceptions import LLMProviderException
from medical_assistant.application.retry_executor import RetryExecutor

from tests.conftest import FakeProvider, FailingProvider


def test_assistant_service_works() -> None:
    assistant_service = AssistantService(provider=FakeProvider(), retry_executor=RetryExecutor())

    assert assistant_service.answer(question="How are you?") == "Testowa odpowiedź na: How are you?"


def test_assistant_service_fails() -> None:
    assistant_service = AssistantService(
        provider=FailingProvider(),
        retry_executor=RetryExecutor(),
    )

    with pytest.raises(LLMProviderException):
        assistant_service.answer(question="How are you?")
