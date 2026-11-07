from dataclasses import asdict
import json
import redis.asyncio as redis

from medical_assistant.application.idempotency_repository import (
    IdempotencyRecord,
    IdempotencyStatus,
)


class RedisIdempotencyRepository:
    def __init__(self) -> None:
        self.client = redis.Redis(host='localhost', port=6379)

    async def get(self, key: str) -> IdempotencyRecord | None:
        result = await self.client.get(key)
        if result:
            data = json.loads(result)
            data["status"] = IdempotencyStatus(data["status"])
            return IdempotencyRecord(**data)
        return None

    async def create_if_not_exists(self, key: str, record: IdempotencyRecord) -> bool:
        result = await self.client.set(
            name=key,
            value=json.dumps(asdict(record)),
            nx=True,
            ex=5 * 60
        )

        return result is not None

    async def update(self, key: str, record: IdempotencyRecord) -> None:
        result = await self.client.set(
            name=key,
            value=json.dumps(asdict(record)),
            xx=True,
            ex=24 * 60 * 60
        )

        if result is None:
            raise KeyError(key)
