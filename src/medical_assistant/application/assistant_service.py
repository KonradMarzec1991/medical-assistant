from medical_assistant.infrastructure.llm.base import LLMProvider


class AssistantService:
    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider

    def answer(self, question: str) -> str:
        return self.provider.generate(user_prompt=question, system_prompt="You are a cardiology assistant")
