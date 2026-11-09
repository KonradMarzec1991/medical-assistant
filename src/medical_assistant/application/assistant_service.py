from medical_assistant.application.dto.llm_request import LLMRequest
from medical_assistant.application.ports.llm_provider import LLMProvider
from medical_assistant.application.retry_executor import RetryExecutor

SYSTEM_INSTRUCTIONS = "You are a cardiology assistant"


class AssistantService:
    def __init__(self, provider: LLMProvider, retry_executor: RetryExecutor) -> None:
        self.provider = provider
        self.retry_executor = retry_executor

    async def answer(self, question: str) -> str:
        return await self.retry_executor.execute(
            lambda: self.provider.generate(request=LLMRequest(instructions=SYSTEM_INSTRUCTIONS, input=question))
        )
