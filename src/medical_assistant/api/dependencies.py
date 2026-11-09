from functools import lru_cache
from fastapi import Request
from medical_assistant.application.assistant_service import AssistantService
from medical_assistant.application.document_processing_service import DocumentProcessingService
from medical_assistant.application.idempotency_service import IdempotencyService
from medical_assistant.application.retry_executor import RetryExecutor
from medical_assistant.config.settings import settings
from medical_assistant.infrastructure.llm.openai_provider import OpenAIProvider
from medical_assistant.infrastructure.repositories.redis_idempotency_repository import RedisIdempotencyRepository


@lru_cache(maxsize=1)
def get_assistant_service() -> AssistantService:
    return AssistantService(
        provider=OpenAIProvider(settings=settings),
        retry_executor=RetryExecutor(max_attempts=3, backoff=1.0),
    )


@lru_cache(maxsize=1)
def get_idempotency_service() -> IdempotencyService:
    repository = RedisIdempotencyRepository()
    return IdempotencyService(repository=repository)


@lru_cache(maxsize=1)
def get_document_processing_service(request: Request) -> DocumentProcessingService:
    return request.app.state.document_processing_service
