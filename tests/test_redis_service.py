from medical_assistant.application.idempotency_repository import IdempotencyRecord, IdempotencyStatus
from medical_assistant.infrastructure.repositories.redis_idempotency_repository import RedisIdempotencyRepository


def test_create_if_not_exists_returns_true_when_key_does_not_exist():
    repository = RedisIdempotencyRepository()
    key = "test:idempotency:12345"

    repository.client.delete(key)

    record = IdempotencyRecord(
        request_hash="12345",
        status=IdempotencyStatus.IN_PROGRESS,
    )

    assert repository.create_if_not_exists(key, record) is True

    repository.client.delete(key)


def test_create_if_not_exists_returns_false_when_key_already_exists():
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