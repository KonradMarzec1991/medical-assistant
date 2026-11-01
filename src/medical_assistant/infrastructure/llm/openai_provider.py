from openai import OpenAI, APITimeoutError, APIConnectionError, AuthenticationError

from medical_assistant.application.exceptions import (
    LLMProviderException,
    LLMTimeoutException,
    LLMConnectionException,
    LLMConfigurationException,
)
from medical_assistant.config.settings import Settings


class OpenAIProvider:
    def __init__(self, settings: Settings):
        self.client = OpenAI(api_key=settings.openai_api_key, timeout=settings.openai_timeout)
        self.model = settings.openai_model

    def generate(self, user_prompt: str, system_prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
        except APITimeoutError as e:
            raise LLMTimeoutException("LLM request timed out") from e
        except APIConnectionError as e:
            raise LLMConnectionException("LLM connection failed") from e
        except AuthenticationError as e:
            raise LLMConfigurationException("LLM authentication failed") from e

        content = response.choices[0].message.content
        if not content:
            raise LLMProviderException("Model returned an empty response")

        return content
