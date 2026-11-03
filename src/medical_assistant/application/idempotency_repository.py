from typing import Protocol, Optional
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


class IdempotencyRepository(Protocol):

    def get(self, key: str) -> IdempotencyRecord | None:
        ...

    def create_if_not_exists(self, key: str, record: IdempotencyRecord) -> bool:
        ...

    def update(self, key: str, record: IdempotencyRecord) -> None:
        ...


class InMemoryIdempotencyRepository:
    def __init__(self) -> None:
        self._records: dict[str, IdempotencyRecord] = {}

    def get(self, key) -> IdempotencyRecord | None:
        return self._records.get(key)

    def create_if_not_exists(self, key: str, record: IdempotencyRecord) -> bool:
        if key in self._records:
            return False

        self._records[key] = record
        return True

    def update(self, key: str, record: IdempotencyRecord) -> None:
        if key not in self._records:
            raise KeyError

        self._records[key] = record
