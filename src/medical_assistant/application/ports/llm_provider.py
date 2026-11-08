from typing import Protocol

from medical_assistant.application.dto.llm_request import LLMRequest
from medical_assistant.application.dto.llm_response import LLMResponse


class LLMProvider(Protocol):
    async def generate(self, request: LLMRequest) -> LLMResponse:
        ...
