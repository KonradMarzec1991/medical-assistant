from unittest.mock import Mock

import pytest
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
)

from medical_assistant.application.exceptions import (
    LLMConfigurationException,
    LLMConnectionException,
    LLMTimeoutException,
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


@pytest.mark.parametrize(
    ("sdk_error", "expected_error"),
    [
        (APITimeoutError(request=Mock()), LLMTimeoutException),
        (APIConnectionError(request=Mock()), LLMConnectionException),
        (make_authentication_error(), LLMConfigurationException),
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