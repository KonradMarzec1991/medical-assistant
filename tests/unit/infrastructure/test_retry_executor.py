from unittest.mock import Mock, patch, MagicMock, call, AsyncMock

import pytest

from medical_assistant.application.exceptions import LLMTimeoutException, LLMConfigurationException
from medical_assistant.application.retry_executor import RetryExecutor


@patch("medical_assistant.application.retry_executor.asyncio.sleep")
async def test_retries_until_success(sleep: MagicMock) -> None:
    mock = AsyncMock(
        side_effect=[
            LLMTimeoutException("Timeout 1"),
            LLMTimeoutException("Timeout 2"),
            "Everything is fine",
        ]
    )

    executor = RetryExecutor()
    result = await executor.execute(mock)

    assert result == "Everything is fine"
    assert mock.call_count == 3

    sleep.assert_has_calls([
        call(1.0),
        call(2.0),
    ])


@patch("medical_assistant.application.retry_executor.asyncio.sleep")
async def test_raises_timeout_after_max_attempts(sleep: MagicMock) -> None:
    mock = AsyncMock(
        side_effect=[
            LLMTimeoutException("Timeout 1"),
            LLMTimeoutException("Timeout 2"),
            LLMTimeoutException("Timeout 3"),
        ]
    )

    executor = RetryExecutor()

    with pytest.raises(LLMTimeoutException):
        await executor.execute(mock)

    assert mock.call_count == 3

    sleep.assert_has_calls([
        call(1.0),
        call(2.0),
    ])


@patch("medical_assistant.application.retry_executor.asyncio.sleep")
async def test_does_not_retry_configuration_error(sleep: MagicMock) -> None:
    mock = AsyncMock(
        side_effect=[
            LLMConfigurationException("LLM Config Exception"),
        ]
    )

    executor = RetryExecutor()

    with pytest.raises(LLMConfigurationException):
        await executor.execute(mock)

    assert mock.call_count == 1

    sleep.assert_not_awaited()


@patch("medical_assistant.application.retry_executor.asyncio.sleep")
async def test_retry_can_execute_operation_twice_after_timeout(sleep):
    operation_count = 0

    def simulate_provider_call():
        nonlocal operation_count

        operation_count += 1

        if operation_count == 1:
            raise LLMTimeoutException("Connection timeout")

        return "Generated answer"

    mock_provider = AsyncMock(side_effect=simulate_provider_call)

    executor = RetryExecutor(max_attempts=2, backoff=1.0)
    result = await executor.execute(mock_provider)

    assert result == "Generated answer"
    assert mock_provider.call_count == 2
    assert operation_count == 2
    sleep.assert_awaited_once_with(1.0)
