import asyncio

from collections.abc import Awaitable, Callable
from typing import TypeVar

from medical_assistant.application.exceptions import LLMTimeoutException

T = TypeVar("T")


class RetryExecutor:
    def __init__(self, max_attempts: int = 3, backoff: float = 1.0) -> None:
        if max_attempts <= 0:
            raise ValueError("max_attempts must be > 0")
        if backoff < 0:
            raise ValueError("backoff must be >= 0")

        self.max_attempts = max_attempts
        self.backoff = backoff

    async def execute(self, func: Callable[[], Awaitable[T]]) -> T:
        for attempt in range(1, self.max_attempts + 1):
            try:
                return await func()
            except LLMTimeoutException:
                if attempt == self.max_attempts:
                    raise

                delay = self.backoff * (2 ** (attempt - 1))
                await asyncio.sleep(delay)
