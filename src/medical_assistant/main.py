import hashlib
from typing import Annotated

from fastapi import Depends, FastAPI, status, HTTPException, Header
from pydantic import BaseModel, Field, ConfigDict

from medical_assistant.api.dependencies import get_assistant_service, get_idempotency_service
from medical_assistant.application.assistant_service import AssistantService
from medical_assistant.application.exceptions import LLMProviderException
from medical_assistant.application.idempotency_service import IdempotencyService

app = FastAPI(title="Enterprise Medical Assistant")


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
    idempotency_service: IdempotencyService = Depends(get_idempotency_service),
    idempotency_key: Annotated[str, Header(alias="Idempotency-Key")] = ...,
) -> AskResponse:
    try:
        answer = idempotency_service.execute(
            key=idempotency_key,
            request_hash=hashlib.sha256(
                request.question.encode("utf-8")
            ).hexdigest(),
            operation=lambda: service.answer(request.question),
        )
        return AskResponse(answer=answer)
    except LLMProviderException:
        raise HTTPException(
            detail="AI service is temporarily unavailable",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )
