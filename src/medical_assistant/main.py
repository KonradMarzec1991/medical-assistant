from functools import lru_cache

from fastapi import Depends, FastAPI, status, HTTPException
from pydantic import BaseModel, Field, ConfigDict

from medical_assistant.application.assistant_service import AssistantService
from medical_assistant.application.exceptions import LLMProviderException
from medical_assistant.infrastructure.llm.openai_provider import OpenAIProvider
from medical_assistant.config.settings import settings
from medical_assistant.infrastructure.utils.retry_executor import RetryExecutor

app = FastAPI(title="Enterprise Medical Assistant")


@lru_cache
def get_assistant_service() -> AssistantService:
    return AssistantService(
        provider=OpenAIProvider(settings=settings),
        retry_executor=RetryExecutor(max_attempts=3, backoff=1.0),
    )


class AskRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    question: str = Field(min_length=1, max_length=2000)


class AskResponse(BaseModel):
    answer: str


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/assistant/ask", response_model=AskResponse)
def ask(
        request: AskRequest,
        service: AssistantService = Depends(get_assistant_service),
) -> AskResponse:
    try:
        answer = service.answer(question=request.question)
        return AskResponse(answer=answer)
    except LLMProviderException:
        raise HTTPException(
            detail="AI service is temporarily unavailable",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE
        )
