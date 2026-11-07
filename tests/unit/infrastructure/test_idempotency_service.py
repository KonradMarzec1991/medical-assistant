from unittest.mock import Mock

from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import pytest

from medical_assistant.application.idempotency_service import IdempotencyService
from medical_assistant.application.idempotency_repository import (
    InMemoryIdempotencyRepository,
    IdempotencyRecord,
    IdempotencyStatus,
)
from medical_assistant.application.exceptions import (
    IdempotencyConflictException,
    IdempotencyInProgressException,
    LLMTimeoutException
)


def test_execute_returns_cached_response():
    repository = InMemoryIdempotencyRepository()
    service = IdempotencyService(repository)

    operation = Mock(return_value="Generated answer")

    first_response = service.execute(
        key="request-123",
        request_hash="hash-abc",
        operation=operation,
    )

    second_response = service.execute(
        key="request-123",
        request_hash="hash-abc",
        operation=operation,
    )

    assert first_response == "Generated answer"
    assert second_response == "Generated answer"
    operation.assert_called_once()


def test_execute_raises_conflict_when_request_hash_differs():
    repository = InMemoryIdempotencyRepository()
    service = IdempotencyService(repository)

    operation = Mock(return_value="Generated answer")

    service.execute(
        key="request-123",
        request_hash="hash-abc",
        operation=operation,
    )

    with pytest.raises(IdempotencyConflictException):
        service.execute(
            key="request-123",
            request_hash="hash-xyz",
            operation=operation,
        )

    operation.assert_called_once()


def test_execute_raises_in_progress_exception():
    repository = InMemoryIdempotencyRepository()
    service = IdempotencyService(repository)

    repository.create_if_not_exists(
        key="request-123",
        record=IdempotencyRecord(
            request_hash="hash-abc",
            status=IdempotencyStatus.IN_PROGRESS,
        ),
    )

    operation = Mock(return_value="Generated answer")

    with pytest.raises(IdempotencyInProgressException):
        service.execute(
            key="request-123",
            request_hash="hash-abc",
            operation=operation,
        )

    operation.assert_not_called()


def test_execute_does_not_mark_record_completed_when_operation_fails():
    repository = InMemoryIdempotencyRepository()
    service = IdempotencyService(repository)

    operation = Mock(side_effect=LLMTimeoutException())

    with pytest.raises(LLMTimeoutException):
        service.execute(
            key="request-123",
            request_hash="hash-abc",
            operation=operation,
        )

    record = repository.get("request-123")

    assert record is not None
    assert record.status == IdempotencyStatus.IN_PROGRESS
    assert record.response is None
    operation.assert_called_once()


class RaceConditionRepository:

    def __init__(self):
        self.get_calls = 0

    def get(self, key: str) -> IdempotencyRecord | None:
        self.get_calls += 1

        if self.get_calls == 1:
            return None

        return IdempotencyRecord(
            request_hash="hash-abc",
            status=IdempotencyStatus.IN_PROGRESS,
        )

    def create_if_not_exists(self, key: str, record: IdempotencyRecord) -> bool:
        return False

    def update(self, key: str, record: IdempotencyRecord,) -> None:
        raise AssertionError("update should not be called")


def test_race_condition():
    repository = RaceConditionRepository()
    service = IdempotencyService(repository)

    operation = Mock(return_value="Generated answer")

    with pytest.raises(IdempotencyInProgressException):
        service.execute(
            key="request-123",
            request_hash="hash-abc",
            operation=operation,
        )

    assert repository.get_calls == 2
    operation.assert_not_called()


class RaceyRepository:

    def __init__(self):
        self._records = {}

    def get(self, key: str) -> IdempotencyRecord | None:
        return self._records.get(key)

    def create_if_not_exists(
        self,
        key: str,
        record: IdempotencyRecord,
    ) -> bool:
        if key in self._records:
            return False

        self._records[key] = record
        return True

    def update(self, key: str, record: IdempotencyRecord) -> None:
        self._records[key] = record


def test_concurrent_requests_execute_operation_only_once():
    repository = RaceyRepository()
    service = IdempotencyService(repository)

    operation_count = 0
    operation_lock = Lock()

    def operation():
        nonlocal operation_count

        with operation_lock:
            operation_count += 1

        return "Generated answer"

    def execute():
        try:
            return service.execute(
                key="request-123",
                request_hash="hash-abc",
                operation=operation,
            )
        except IdempotencyInProgressException:
            return "IN_PROGRESS"

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(execute),
            executor.submit(execute),
        ]

        results = [future.result() for future in futures]

    assert operation_count == 1