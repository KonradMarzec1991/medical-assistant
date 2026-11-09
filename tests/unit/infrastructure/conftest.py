from unittest.mock import Mock, AsyncMock

import pytest
from _pytest.monkeypatch import MonkeyPatch
from fastapi.testclient import TestClient

from medical_assistant.api.dependencies import get_idempotency_service
from medical_assistant.application.assistant_service import AssistantService
from medical_assistant.application.exceptions import LLMProviderException, LLMTimeoutException
from medical_assistant.application.idempotency_repository import InMemoryIdempotencyRepository
from medical_assistant.application.idempotency_service import IdempotencyService
from medical_assistant.application.retry_executor import RetryExecutor
from medical_assistant.application.dto.llm_response import LLMResponse
from medical_assistant.application.dto.llm_request import LLMRequest

from medical_assistant.main import app, get_assistant_service


class FakeProvider:
    def __init__(self):
        self.system_prompt = None
        self.user_prompt = None

    async def generate(self, request: LLMRequest) -> LLMResponse[str]:
        self.system_prompt = request.instructions
        self.user_prompt = request.input

        return LLMResponse(content=f"Testowa odpowiedź na: {request.input}", input_tokens=None, output_tokens=None)


class FailingProvider:
    async def generate(self, request: LLMRequest) -> LLMResponse[str]:
        raise LLMProviderException("Provider failed")


@pytest.fixture
def assistant_service() -> AssistantService:
    return AssistantService(
        provider=FakeProvider(),
        retry_executor=RetryExecutor(),
        model="some-model"
    )


@pytest.fixture
def idempotency_service() -> IdempotencyService:
    return IdempotencyService(repository=InMemoryIdempotencyRepository())


@pytest.fixture
def client(assistant_service: AssistantService, idempotency_service: IdempotencyService):
    app.dependency_overrides[get_assistant_service] = lambda: assistant_service
    app.dependency_overrides[get_idempotency_service] = lambda: idempotency_service
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.pop(get_assistant_service, None)
        app.dependency_overrides.pop(idempotency_service, None)


@pytest.fixture
def failing_client(idempotency_service: IdempotencyService):
    service = AssistantService(
        provider=FailingProvider(),
        retry_executor=RetryExecutor(),
        model="some-model"
    )
    app.dependency_overrides[get_assistant_service] = lambda: service
    app.dependency_overrides[get_idempotency_service] = lambda: idempotency_service
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.pop(get_assistant_service, None)
        app.dependency_overrides.pop(idempotency_service, None)


@pytest.fixture
def retry_client(monkeypatch: MonkeyPatch):
    provider = FakeProvider()
    provider.generate = AsyncMock(
        side_effect=[
            LLMTimeoutException("Timeout 1"),
            LLMTimeoutException("Timeout 2"),
            LLMResponse(content="Everything is fine", input_tokens=None, output_tokens=None),
        ]
    )

    service = AssistantService(
        provider=provider,
        retry_executor=RetryExecutor(),
        model="some-model"
    )

    monkeypatch.setattr(
        "medical_assistant.application.retry_executor.asyncio.sleep",
        AsyncMock()
    )

    app.dependency_overrides[get_assistant_service] = lambda: service
    try:
        with TestClient(app) as test_client:
            yield test_client, provider
    finally:
        app.dependency_overrides.pop(get_assistant_service, None)
