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

from medical_assistant.main import app, get_assistant_service


class FakeProvider:
    def __init__(self):
        self.system_prompt = None
        self.user_prompt = None

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt

        return f"Testowa odpowiedź na: {user_prompt}"


class FailingProvider:
    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        raise LLMProviderException("Provider failed")


@pytest.fixture
def assistant_service() -> AssistantService:
    return AssistantService(
        provider=FakeProvider(),
        retry_executor=RetryExecutor(),
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
            "Everything is fine",
        ]
    )

    service = AssistantService(
        provider=provider,
        retry_executor=RetryExecutor(),
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
