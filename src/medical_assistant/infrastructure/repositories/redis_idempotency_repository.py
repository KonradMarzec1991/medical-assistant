from dataclasses import asdict
import json
import redis

from medical_assistant.application.idempotency_repository import (
    IdempotencyRepository,
    IdempotencyRecord, IdempotencyStatus,
)



class RedisIdempotencyRepository:
    def __init__(self) -> None:
        self.client = redis.Redis(host='localhost', port=6379)

    def get(self, key: str) -> IdempotencyRecord | None:
        result = self.client.get(key)
        if result:
            data = json.loads(result)
            data["status"] = IdempotencyStatus(data["status"])
            return IdempotencyRecord(**data)
        return None

    def create_if_not_exists(self, key: str, record: IdempotencyRecord) -> str | None:
        result = self.client.set(
            name=key,
            value=json.dumps(asdict(record)),
            nx=True
        )

        return result is not None
