from threading import Lock
from typing import Protocol
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
        self._lock = Lock()

    def get(self, key) -> IdempotencyRecord | None:
        with self._lock:
            return self._records.get(key)

    def create_if_not_exists(self, key: str, record: IdempotencyRecord) -> bool:
        with self._lock:
            if key in self._records:
                return False

            self._records[key] = record
            return True

    def update(self, key: str, record: IdempotencyRecord) -> None:
        with self._lock:
            if key not in self._records:
                raise KeyError

            self._records[key] = record
