from collections.abc import Callable

from medical_assistant.application.exceptions import (
    IdempotencyConflictException,
    IdempotencyInProgressException,
    IdempotencyReservationException,
)

from medical_assistant.application.idempotency_repository import (
    IdempotencyRepository,
    IdempotencyRecord,
    IdempotencyStatus,
)


class IdempotencyService:
    def __init__(self, repository: IdempotencyRepository) -> None:
        self.repository = repository

    def execute(
        self,
        key: str,
        request_hash: str,
        operation: Callable[[], str],
    ) -> str:

        # 1. Check whether the key already exists.
        record = self.repository.get(key)

        if record is not None:
            return self._handle_existing_record(record, request_hash)

        # 2. Try to reserve the key.
        created = self.repository.create_if_not_exists(
            key=key,
            record=IdempotencyRecord(
                request_hash=request_hash,
                status=IdempotencyStatus.IN_PROGRESS,
            ),
        )

        # 3. Another request may have reserved the key.
        if not created:
            record = self.repository.get(key)

            if record is None:
                raise IdempotencyReservationException()

            return self._handle_existing_record(record, request_hash)

        # 4. Execute the operation.
        response = operation()

        # 5. Store the result.
        self.repository.update(
            key=key,
            record=IdempotencyRecord(
                request_hash=request_hash,
                status=IdempotencyStatus.COMPLETED,
                response=response,
            ),
        )

        # 6. Return the result.
        return response

    @staticmethod
    def _handle_existing_record(
        record: IdempotencyRecord,
        request_hash: str,
    ) -> str:

        if record.request_hash != request_hash:
            raise IdempotencyConflictException()

        if record.status == IdempotencyStatus.COMPLETED:
            if record.response is None:
                raise RuntimeError("Completed record has no response")

            return record.response

        if record.status == IdempotencyStatus.IN_PROGRESS:
            raise IdempotencyInProgressException()

        raise RuntimeError(f"Unsupported idempotency status: {record.status}")