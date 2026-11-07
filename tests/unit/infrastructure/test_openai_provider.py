from unittest.mock import Mock, AsyncMock

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
        response=AsyncMock(status_code=401, headers={}, request=Mock()),
        body=None,
    )


def make_rate_limit_error() -> RateLimitError:
    return RateLimitError(
        "Rate limit exceeded",
        response=AsyncMock(status_code=401, headers={}, request=Mock()),
        body=None,
    )


def make_api_status_error() -> APIStatusError:
    return APIStatusError(
        message="LLM API returned an error",
        response=AsyncMock(status_code=500, headers={}, request=Mock()),
        body=None,
    )


@pytest.mark.parametrize(
    ("sdk_error", "expected_error"),
    [
        (APITimeoutError(request=AsyncMock()), LLMTimeoutException),
        (APIConnectionError(request=AsyncMock()), LLMConnectionException),
        (make_authentication_error(), LLMConfigurationException),
        (make_rate_limit_error(), LLMRateLimitException),
        (make_api_status_error(), LLMResponseException),
    ],
)
async def test_generate_maps_openai_errors(
    sdk_error: Exception,
    expected_error: type[Exception],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = make_provider()
    create = AsyncMock(side_effect=sdk_error)
    monkeypatch.setattr(provider.client.chat.completions, "create", create)

    with pytest.raises(expected_error) as exc_info:
        await provider.generate(user_prompt="Question", system_prompt="Instructions")

    assert exc_info.value.__cause__ is sdk_error
    create.assert_awaited_once()


async def test_generate_returns_response_content(monkeypatch):
    provider = make_provider()

    response = AsyncMock(choices=[AsyncMock(message=AsyncMock(content="Generated answer"))])

    create = AsyncMock(return_value=response)
    monkeypatch.setattr(provider.client.chat.completions, "create", create)

    result = await provider.generate(
        user_prompt="Question",
        system_prompt="Instructions",
    )

    assert result == "Generated answer"
    create.assert_awaited_once()


async def test_generate_raises_when_response_is_empty(monkeypatch):
    provider = make_provider()

    response = AsyncMock(choices=[AsyncMock(message=AsyncMock(content=""))])

    create = AsyncMock(return_value=response)
    monkeypatch.setattr(provider.client.chat.completions, "create", create)

    with pytest.raises(LLMResponseException):
        await provider.generate(
            user_prompt="Question",
            system_prompt="Instructions",
        )
