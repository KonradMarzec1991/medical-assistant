from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError, RateLimitError, APIStatusError,
)

from medical_assistant.application.exceptions import (
    LLMConfigurationException,
    LLMConnectionException,
    LLMTimeoutException, LLMRateLimitException, LLMResponseException,
)
from medical_assistant.config.settings import Settings
from medical_assistant.infrastructure.llm.openai_provider import OpenAIProvider


def make_provider() -> OpenAIProvider:
    return OpenAIProvider(
        settings=Settings(
            openai_api_key="test-key",
            openai_model="test-model",
            openai_timeout=1,
        )
    )

def make_authentication_error() -> AuthenticationError:
    return AuthenticationError(
        "Unauthorized",
        response=Mock(status_code=401, headers={}, request=Mock()),
        body=None,
    )


def make_rate_limit_error() -> RateLimitError:
    return RateLimitError(
        "Rate limit exceeded",
        response=Mock(status_code=401, headers={}, request=Mock()),
        body=None,
    )


def make_api_status_error() -> APIStatusError:
    return APIStatusError(
        message="LLM API returned an error",
        response=Mock(status_code=500, headers={}, request=Mock()),
        body=None,
    )


@pytest.mark.parametrize(
    ("sdk_error", "expected_error"),
    [
        (APITimeoutError(request=Mock()), LLMTimeoutException),
        (APIConnectionError(request=Mock()), LLMConnectionException),
        (make_authentication_error(), LLMConfigurationException),
        (make_rate_limit_error(), LLMRateLimitException),
        (make_api_status_error(), LLMResponseException),
    ],
)
def test_generate_maps_openai_errors(
    sdk_error: Exception,
    expected_error: type[Exception],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = make_provider()
    create = Mock(side_effect=sdk_error)
    monkeypatch.setattr(provider.client.chat.completions, "create", create)

    with pytest.raises(expected_error) as exc_info:
        provider.generate(user_prompt="Question", system_prompt="Instructions")

    assert exc_info.value.__cause__ is sdk_error
    create.assert_called_once()


def test_generate_returns_response_content(monkeypatch):
    provider = make_provider()

    response = SimpleNamespace(choices=[Mock(message=Mock(content="Generated answer"))])

    create = Mock(return_value=response)
    monkeypatch.setattr(provider.client.chat.completions, "create", create)

    result = provider.generate(
        user_prompt="Question",
        system_prompt="Instructions",
    )

    assert result == "Generated answer"
    create.assert_called_once()


def test_generate_raises_when_response_is_empty(monkeypatch):
    provider = make_provider()

    response = SimpleNamespace(choices=[Mock(message=Mock(content=""))])

    create = Mock(return_value=response)
    monkeypatch.setattr(provider.client.chat.completions, "create", create)

    with pytest.raises(LLMResponseException):
        provider.generate(
            user_prompt="Question",
            system_prompt="Instructions",
        )
