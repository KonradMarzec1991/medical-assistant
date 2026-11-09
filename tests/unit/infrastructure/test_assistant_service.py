import pytest

from medical_assistant.application.assistant_service import AssistantService
from medical_assistant.application.exceptions import LLMProviderException
from medical_assistant.application.retry_executor import RetryExecutor

from tests.unit.infrastructure.conftest import FakeProvider, FailingProvider


async def test_assistant_service_works() -> None:
    assistant_service = AssistantService(provider=FakeProvider(), retry_executor=RetryExecutor(), model="some-model")

    assert await assistant_service.answer(question="How are you?") == "Testowa odpowiedź na: How are you?"


async def test_assistant_service_fails() -> None:
    assistant_service = AssistantService(
        provider=FailingProvider(),
        retry_executor=RetryExecutor(),
        model="some-model"
    )

    with pytest.raises(LLMProviderException):
        await assistant_service.answer(question="How are you?")
