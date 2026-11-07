import asyncio
from unittest.mock import Mock, AsyncMock

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


async def test_execute_returns_cached_response():
    repository = InMemoryIdempotencyRepository()
    service = IdempotencyService(repository)

    operation = AsyncMock(return_value="Generated answer")

    first_response = await service.execute(
        key="request-123",
        request_hash="hash-abc",
        operation=operation,
    )

    second_response = await service.execute(
        key="request-123",
        request_hash="hash-abc",
        operation=operation,
    )

    assert first_response == "Generated answer"
    assert second_response == "Generated answer"
    operation.assert_awaited_once()


async def test_execute_raises_conflict_when_request_hash_differs():
    repository = InMemoryIdempotencyRepository()
    service = IdempotencyService(repository)

    operation = AsyncMock(return_value="Generated answer")

    await service.execute(
        key="request-123",
        request_hash="hash-abc",
        operation=operation,
    )

    with pytest.raises(IdempotencyConflictException):
        await service.execute(
            key="request-123",
            request_hash="hash-xyz",
            operation=operation,
        )

    operation.assert_awaited_once()


async def test_execute_raises_in_progress_exception():
    repository = InMemoryIdempotencyRepository()
    service = IdempotencyService(repository)

    await repository.create_if_not_exists(
        key="request-123",
        record=IdempotencyRecord(
            request_hash="hash-abc",
            status=IdempotencyStatus.IN_PROGRESS,
        ),
    )

    operation = AsyncMock(return_value="Generated answer")

    with pytest.raises(IdempotencyInProgressException):
        await service.execute(
            key="request-123",
            request_hash="hash-abc",
            operation=operation,
        )

    operation.assert_not_awaited()


async def test_execute_does_not_mark_record_completed_when_operation_fails():
    repository = InMemoryIdempotencyRepository()
    service = IdempotencyService(repository)

    operation = AsyncMock(side_effect=LLMTimeoutException())

    with pytest.raises(LLMTimeoutException):
        await service.execute(
            key="request-123",
            request_hash="hash-abc",
            operation=operation,
        )

    record = await repository.get("request-123")

    assert record is not None
    assert record.status == IdempotencyStatus.IN_PROGRESS
    assert record.response is None
    operation.assert_awaited_once()


class RaceConditionRepository:

    def __init__(self):
        self.get_calls = 0

    async def get(self, key: str) -> IdempotencyRecord | None:
        self.get_calls += 1

        if self.get_calls == 1:
            return None

        return IdempotencyRecord(
            request_hash="hash-abc",
            status=IdempotencyStatus.IN_PROGRESS,
        )

    async def create_if_not_exists(self, key: str, record: IdempotencyRecord) -> bool:
        return False

    async def update(self, key: str, record: IdempotencyRecord,) -> None:
        raise AssertionError("update should not be called")


async def test_race_condition():
    repository = RaceConditionRepository()
    service = IdempotencyService(repository)

    operation = AsyncMock(return_value="Generated answer")

    with pytest.raises(IdempotencyInProgressException):
        await service.execute(
            key="request-123",
            request_hash="hash-abc",
            operation=operation,
        )

    assert repository.get_calls == 2
    operation.assert_not_awaited()


class RaceyRepository:

    def __init__(self):
        self._records = {}

    async def get(self, key: str) -> IdempotencyRecord | None:
        return self._records.get(key)

    async def create_if_not_exists(
        self,
        key: str,
        record: IdempotencyRecord,
    ) -> bool:
        if key in self._records:
            return False

        self._records[key] = record
        return True

    async def update(self, key: str, record: IdempotencyRecord) -> None:
        self._records[key] = record


async def test_concurrent_requests_execute_operation_only_once():
    repository = RaceyRepository()
    service = IdempotencyService(repository)

    operation_count = 0
    operation_lock = Lock()

    async def operation():
        nonlocal operation_count
        with operation_lock:
            operation_count += 1

        return "Generated answer"

    async def execute():
        try:
            return await service.execute(
                key="request-123",
                request_hash="hash-abc",
                operation=operation,
            )
        except IdempotencyInProgressException:
            return "IN_PROGRESS"

    await asyncio.gather(execute(), execute())

    assert operation_count == 1
