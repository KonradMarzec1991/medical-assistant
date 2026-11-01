from fastapi import FastAPI
from pydantic import BaseModel, Field, ConfigDict

from medical_assisstant.application.assistant_service import AssistantService
from medical_assisstant.infrastructure.llm.openai_provider import OpenAIProvider
from medical_assisstant.config.settings import settings

app = FastAPI(title="Enterprise Medical Assistant")

assistant_service = AssistantService(provider=OpenAIProvider(settings=settings))


class AskRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    question: str = Field(min_length=1, max_length=2000)


class AskResponse(BaseModel):
    answer: str


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/assistant/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    answer = assistant_service.answer(question=request.question)
    return AskResponse(answer=answer)
