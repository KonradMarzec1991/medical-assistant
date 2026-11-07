import asyncio
import time
from enum import Enum
from typing import Callable
from uuid import uuid4


class CircuitState(Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout: float = 5.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")

        if recovery_timeout <= 0:
            raise ValueError("recovery_timeout must be > 0")

        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.clock = clock

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.state_changed_at = self.clock()

        self._half_open_trial_id: str | None = None
        self._lock = asyncio.Lock()

    async def allow_request(self) -> str | None:
        async with self._lock:
            now = self.clock()

            if self.state == CircuitState.CLOSED:
                return uuid4().hex

            if self.state == CircuitState.OPEN:
                elapsed = now - self.state_changed_at

                if elapsed < self.recovery_timeout:
                    return None

                self.state = CircuitState.HALF_OPEN

            if self.state == CircuitState.HALF_OPEN:
                if self._half_open_trial_id is not None:
                    return None

                trial_id = uuid4().hex
                self._half_open_trial_id = trial_id

                return trial_id

            return None

    async def record_success(self, request_id: str) -> None:
        async with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                if request_id != self._half_open_trial_id:
                    return

                self._half_open_trial_id = None
                self.failure_count = 0
                self.state = CircuitState.CLOSED
                self.state_changed_at = self.clock()
                return

            if self.state == CircuitState.CLOSED:
                self.failure_count = 0

    async def record_failure(self, request_id: str) -> None:
        async with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                if request_id != self._half_open_trial_id:
                    return

                self._half_open_trial_id = None
                self.state = CircuitState.OPEN
                self.state_changed_at = self.clock()
                return

            if self.state == CircuitState.CLOSED:
                self.failure_count += 1

                if self.failure_count >= self.failure_threshold:
                    self.state = CircuitState.OPEN
                    self.state_changed_at = self.clock()