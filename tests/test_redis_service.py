import pytest

from medical_assistant.application.idempotency_repository import IdempotencyRecord, IdempotencyStatus
from medical_assistant.infrastructure.repositories.redis_idempotency_repository import RedisIdempotencyRepository


def test_create_if_not_exists_returns_true_when_key_does_not_exist() -> None:
    repository = RedisIdempotencyRepository()
    key = "test:idempotency:12345"

    repository.client.delete(key)

    record = IdempotencyRecord(
        request_hash="12345",
        status=IdempotencyStatus.IN_PROGRESS,
    )

    assert repository.create_if_not_exists(key, record) is True
    assert repository.client.ttl(key) == 300

    repository.client.delete(key)


def test_create_if_not_exists_returns_false_when_key_already_exists() -> None:
    repository = RedisIdempotencyRepository()
    key = "test:idempotency:12345"

    repository.client.delete(key)

    record = IdempotencyRecord(
        request_hash="12345",
        status=IdempotencyStatus.IN_PROGRESS,
    )

    repository.create_if_not_exists(key, record)

    assert repository.create_if_not_exists(key, record) is False

    repository.client.delete(key)


def test_get_returns_record_when_key_exists() -> None:
    repository = RedisIdempotencyRepository()
    key = "test:idempotency:get"

    repository.client.delete(key)

    record = IdempotencyRecord(
        request_hash="12345",
        status=IdempotencyStatus.IN_PROGRESS,
    )

    repository.create_if_not_exists(key, record)

    result = repository.get(key)

    assert isinstance(result, IdempotencyRecord)
    assert isinstance(result.status, IdempotencyStatus)
    assert result.request_hash == "12345"
    assert result.status == IdempotencyStatus.IN_PROGRESS
    assert result.response is None


def test_get_returns_none_when_key_does_not_exist() -> None:
    repository = RedisIdempotencyRepository()
    key = "test:idempotency:missing"

    repository.client.delete(key)

    result = repository.get(key)

    assert result is None


def test_update_updates_existing_record() -> None:
    repository = RedisIdempotencyRepository()
    key = "test:idempotency:update"

    repository.client.delete(key)

    initial_record = IdempotencyRecord(
        request_hash="12345",
        status=IdempotencyStatus.IN_PROGRESS,
    )

    repository.create_if_not_exists(key, initial_record)

    updated_record = IdempotencyRecord(
        request_hash="12345",
        status=IdempotencyStatus.COMPLETED,
        response="Generated answer",
    )

    repository.update(key, updated_record)

    result = repository.get(key)

    assert result == updated_record
    assert repository.client.ttl(key) == 86400

    repository.client.delete(key)


def test_update_raises_key_error_when_key_does_not_exist() -> None:
    repository = RedisIdempotencyRepository()

    with pytest.raises(KeyError):
        repository.update(
            key="not:exist:key",
            record=IdempotencyRecord(request_hash="12345", status=IdempotencyStatus.IN_PROGRESS)
        )
