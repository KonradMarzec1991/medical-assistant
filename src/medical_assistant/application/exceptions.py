

class LLMProviderException(Exception):
    pass


class LLMConnectionException(LLMProviderException):
    pass


class LLMTimeoutException(LLMProviderException):
    pass


class LLMConfigurationException(LLMProviderException):
    pass


class LLMRateLimitException(LLMProviderException):
    pass


class LLMResponseException(LLMProviderException):
    pass


class LLMIncompleteResponseException(LLMProviderException):
    pass


class IdempotencyConflictException(Exception):
    """Same idempotency key used with a different request."""


class IdempotencyInProgressException(Exception):
    """An operation with this key is still running."""


class IdempotencyReservationException(Exception):
    """Could not reserve the idempotency key."""