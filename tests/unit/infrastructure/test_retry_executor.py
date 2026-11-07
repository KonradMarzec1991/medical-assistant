from unittest.mock import Mock, patch, MagicMock, call

import pytest

from medical_assistant.application.exceptions import LLMTimeoutException, LLMConfigurationException
from medical_assistant.application.retry_executor import RetryExecutor


@patch("medical_assistant.application.retry_executor.time.sleep")
def test_retries_until_success(sleep: MagicMock) -> None:
    mock = Mock(
        side_effect=[
            LLMTimeoutException("Timeout 1"),
            LLMTimeoutException("Timeout 2"),
            "Everything is fine",
        ]
    )

    executor = RetryExecutor()
    result = executor.execute(mock)

    assert result == "Everything is fine"
    assert mock.call_count == 3

    sleep.assert_has_calls([
        call(1.0),
        call(2.0),
    ])


@patch("medical_assistant.application.retry_executor.time.sleep")
def test_raises_timeout_after_max_attempts(sleep: MagicMock) -> None:
    mock = Mock(
        side_effect=[
            LLMTimeoutException("Timeout 1"),
            LLMTimeoutException("Timeout 2"),
            LLMTimeoutException("Timeout 3"),
        ]
    )

    executor = RetryExecutor()

    with pytest.raises(LLMTimeoutException):
        executor.execute(mock)

    assert mock.call_count == 3

    sleep.assert_has_calls([
        call(1.0),
        call(2.0),
    ])


@patch("medical_assistant.application.retry_executor.time.sleep")
def test_does_not_retry_configuration_error(sleep: MagicMock) -> None:
    mock = Mock(
        side_effect=[
            LLMConfigurationException("LLM Config Exception"),
        ]
    )

    executor = RetryExecutor()

    with pytest.raises(LLMConfigurationException):
        executor.execute(mock)

    assert mock.call_count == 1

    sleep.assert_not_called()


@patch("medical_assistant.application.retry_executor.time.sleep")
def test_retry_can_execute_operation_twice_after_timeout(sleep):
    operation_count = 0

    def simulate_provider_call():
        nonlocal operation_count

        operation_count += 1

        if operation_count == 1:
            raise LLMTimeoutException("Connection timeout")

        return "Generated answer"

    mock_provider = Mock(side_effect=simulate_provider_call)

    executor = RetryExecutor(max_attempts=2, backoff=1.0)
    result = executor.execute(mock_provider)

    assert result == "Generated answer"
    assert mock_provider.call_count == 2
    assert operation_count == 2
    sleep.assert_called_once_with(1.0)