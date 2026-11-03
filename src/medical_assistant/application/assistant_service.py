from typing import Protocol, Optional

from medical_assistant.infrastructure.llm.base import LLMProvider
from medical_assistant.infrastructure.utils.retry_executor import RetryExecutor


from dataclasses import dataclass
from enum import Enum


class IdempotencyStatus(str, Enum):
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"


@dataclass
class IdempotencyRecord:
    request_hash: str
    status: IdempotencyStatus
    response: str | None = None


class IdempotencyService:
    def __init__(self, key: str) -> None:
        self.key = key


class IdempotencyRepository(Protocol):

    def get(self, key: str) -> IdempotencyRecord | None:
        ...

    def create_if_not_exists(self, key: str, record: IdempotencyRecord,) -> bool:
        ...

    def update(self, key: str, record: IdempotencyRecord,) -> None:
        ...

class InMemoryIdempotencyRepository:
    keys = {}

    def __init__(self, repository: IdempotencyRepository) -> None:
        self.repository = repository


    def get(self, key) -> Optional[str]:
        return self.keys.get(key)

    def set(self, key: str, value: str) -> None:
        self.keys[key] = value


class AssistantService:
    def __init__(self, provider: LLMProvider, retry_executor: RetryExecutor) -> None:
        self.provider = provider
        self.retry_executor = retry_executor

    def answer(self, question: str) -> str:
        return self.retry_executor.execute(
            lambda: self.provider.generate(user_prompt=question, system_prompt="You are a cardiology assistant")
        )
