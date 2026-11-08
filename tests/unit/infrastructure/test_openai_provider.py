from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    RateLimitError,
)

from medical_assistant.application.dto.llm_request import LLMRequest
from medical_assistant.application.dto.llm_response import LLMResponse
from medical_assistant.application.exceptions import (
    LLMConfigurationException,
    LLMConnectionException,
    LLMRateLimitException,
    LLMResponseException,
    LLMTimeoutException,
)
from medical_assistant.infrastructure.llm.openai_provider import OpenAIProvider


def make_provider() -> OpenAIProvider:
    client = Mock()
    client.responses = Mock()
    client.responses.create = AsyncMock()
    return OpenAIProvider(client=client)


def make_request() -> LLMRequest:
    return LLMRequest(
        instructions="You are a medical assistant",
        input="What is blood pressure?",
        model="test-model",
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
        response=Mock(status_code=429, headers={}, request=Mock()),
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
async def test_generate_maps_openai_errors(
    sdk_error: Exception,
    expected_error: type[Exception],
) -> None:
    provider = make_provider()
    create = AsyncMock(side_effect=sdk_error)
    provider.client.responses.create = create

    with pytest.raises(expected_error) as exc_info:
        await provider.generate(make_request())

    assert exc_info.value.__cause__ is sdk_error
    create.assert_awaited_once_with(
        model="test-model",
        instructions="You are a medical assistant",
        input="What is blood pressure?",
    )


async def test_generate_returns_response_content_and_usage() -> None:
    provider = make_provider()
    response = SimpleNamespace(
        status="completed",
        output_text="Generated answer",
        incomplete_details=None,
        usage=SimpleNamespace(input_tokens=12, output_tokens=7),
    )
    provider.client.responses.create.return_value = response

    result = await provider.generate(make_request())

    assert result == LLMResponse(
        content="Generated answer",
        input_tokens=12,
        output_tokens=7,
    )
    provider.client.responses.create.assert_awaited_once()


async def test_generate_raises_when_response_is_empty() -> None:
    provider = make_provider()
    provider.client.responses.create.return_value = SimpleNamespace(
        status="completed",
        output_text="",
        incomplete_details=None,
        usage=None,
    )

    with pytest.raises(LLMResponseException, match="empty response"):
        await provider.generate(make_request())
