

class LLMProviderException(Exception):
    pass


class LLMConnectionException(Exception):
    pass


class LLMTimeoutException(Exception):
    pass


class LLMConfigurationException(Exception):
    pass


class IdempotencyConflictException(Exception):
    """Same idempotency key used with a different request."""


class IdempotencyInProgressException(Exception):
    """An operation with this key is still running."""


class IdempotencyReservationException(Exception):
    """Could not reserve the idempotency key."""