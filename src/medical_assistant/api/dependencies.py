from functools import lru_cache
from fastapi import Request
from medical_assistant.application.assistant_service import AssistantService
from medical_assistant.application.document_processing_service import DocumentProcessingService
from medical_assistant.application.idempotency_service import IdempotencyService

from medical_assistant.infrastructure.repositories.redis_idempotency_repository import RedisIdempotencyRepository


def get_assistant_service(request: Request) -> AssistantService:
    return request.app.state.assistant_service


def get_document_processing_service(
    request: Request,
) -> DocumentProcessingService:
    return request.app.state.document_processing_service


@lru_cache(maxsize=1)
def get_idempotency_service() -> IdempotencyService:
    repository = RedisIdempotencyRepository()
    return IdempotencyService(repository=repository)

