from openai import OpenAI

from medical_assisstant.config.settings import Settings


class LLMProviderException(Exception):
    pass


class OpenAIProvider:
    def __init__(self, settings: Settings):
        self.client = OpenAI(api_key=settings.openai_api_key, timeout=settings.openai_timeout)
        self.model = settings.openai_model

    def generate(self, user_prompt: str, system_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        content = response.choices[0].message.content
        if not content:
            raise LLMProviderException("Model returned an empty response")

        return content
