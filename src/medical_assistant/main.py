from functools import lru_cache

from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field, ConfigDict

from medical_assistant.application.assistant_service import AssistantService
from medical_assistant.infrastructure.llm.openai_provider import OpenAIProvider
from medical_assistant.config.settings import settings

app = FastAPI(title="Enterprise Medical Assistant")


@lru_cache
def get_assistant_service() -> AssistantService:
    return AssistantService(provider=OpenAIProvider(settings=settings))


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
    answer = service.answer(question=request.question)
    return AskResponse(answer=answer)
