from medical_assistant.infrastructure.llm.base import LLMProvider
from medical_assistant.infrastructure.utils.retry_executor import RetryExecutor


class AssistantService:
    def __init__(self, provider: LLMProvider, retry_executor: RetryExecutor) -> None:
        self.provider = provider
        self.retry_executor = retry_executor

    def answer(self, question: str) -> str:
        return self.retry_executor.execute(
            lambda: self.provider.generate(user_prompt=question, system_prompt="You are a cardiology assistant")
        )
