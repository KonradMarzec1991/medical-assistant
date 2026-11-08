from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    RateLimitError,
)

from medical_assistant.application.dto.llm_request import LLMRequest
from medical_assistant.application.dto.llm_response import LLMResponse
from medical_assistant.application.exceptions import (
    LLMTimeoutException,
    LLMConnectionException,
    LLMConfigurationException,
    LLMRateLimitException,
    LLMResponseException, LLMIncompleteResponseException,
)


class OpenAIProvider:
    def __init__(self, client: AsyncOpenAI):
        self.client = client

    async def generate(self, request: LLMRequest) -> LLMResponse:
        params = {
            "model": request.model,
            "instructions": request.instructions,
            "input": request.input,
        }

        if request.max_output_tokens is not None:
            params["max_output_tokens"] = request.max_output_tokens

        if request.temperature is not None:
            params["temperature"] = request.temperature

        try:
            if request.output_type is str:
                response = await self.client.responses.create(**params)
                content = response.output_text
            else:
                response = await self.client.responses.parse(
                    **params,
                    text_format=request.output_type,
                )
                content = response.output_parsed

        except APITimeoutError as e:
            raise LLMTimeoutException(
                "LLM request timed out"
            ) from e

        except APIConnectionError as e:
            raise LLMConnectionException(
                "LLM connection failed"
            ) from e

        except AuthenticationError as e:
            raise LLMConfigurationException(
                "LLM authentication failed"
            ) from e

        except RateLimitError as e:
            raise LLMRateLimitException(
                "LLM rate limit exceeded"
            ) from e

        except APIStatusError as e:
            raise LLMResponseException(
                "LLM API returned an error"
            ) from e

        if response.status == "incomplete":
            reason = (
                response.incomplete_details.reason
                if response.incomplete_details
                else "unknown"
            )

            raise LLMIncompleteResponseException(
                f"LLM response incomplete: {reason}"
            )

        if response.status != "completed":
            raise LLMResponseException(
                f"Unexpected LLM response status: {response.status}"
            )

        if content is None:
            raise LLMResponseException(
                "Model did not return a valid response"
            )

        if isinstance(content, str) and not content.strip():
            raise LLMResponseException(
                "Model returned an empty text response"
            )

        usage = response.usage

        return LLMResponse(
            content=content,
            input_tokens=usage.input_tokens if usage else None,
            output_tokens=usage.output_tokens if usage else None,
        )
