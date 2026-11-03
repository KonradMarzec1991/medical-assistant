import time
from collections.abc import Callable
from typing import TypeVar

from medical_assistant.application.exceptions import LLMTimeoutException

T = TypeVar("T")


class RetryExecutor:
    def __init__(self, max_attempts: int, backoff: float = 1.0) -> None:
        self.max_attempts = max_attempts
        self.backoff = backoff

    def execute(self, func: Callable[[], T]) -> T:
        for attempt in range(1, self.max_attempts + 1):
            try:
                return func()
            except LLMTimeoutException as e:
                if attempt == self.max_attempts:
                    raise

                delay = self.backoff * (2 ** (attempt - 1))
                time.sleep(delay)

