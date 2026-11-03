from unittest.mock import Mock, patch, MagicMock, call

import pytest

from medical_assistant.application.exceptions import LLMTimeoutException, LLMConfigurationException
from medical_assistant.infrastructure.utils.retry_executor import RetryExecutor


@patch("medical_assistant.infrastructure.utils.retry_executor.time.sleep")
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


@patch("medical_assistant.infrastructure.utils.retry_executor.time.sleep")
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


@patch("medical_assistant.infrastructure.utils.retry_executor.time.sleep")
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