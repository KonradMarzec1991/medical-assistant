from enum import Enum

from pydantic import BaseModel, ConfigDict

from medical_assistant.application.dto.llm_request import LLMRequest
from medical_assistant.application.exceptions import LLMResponseException
from medical_assistant.application.ports.llm_provider import LLMProvider


class MedicalDocumentType(str, Enum):
    laboratory_results = "laboratory_results"
    prescription = "prescription"
    hospital_discharge = "hospital_discharge"
    other = "other"


class MedicalDocument(BaseModel):
    model_config = ConfigDict(strict=True)

    is_medical_document: bool
    document_type: MedicalDocumentType
    patient_name: str | None
    document_date: str | None


class DocumentClassifier:
    def __init__(
        self,
        provider: LLMProvider,
        model: str,
    ) -> None:
        self.provider = provider
        self.model = model

    async def classify(self, document_text: str) -> MedicalDocument:
        instructions = """
            You are a medical document information extraction assistant.

            Extract information from the provided document.

            Never invent missing information.
            Use null when information is missing.

            If the input is not a medical document:
            - is_medical_document must be false
            - document_type must be other
        """

        request = LLMRequest(
            instructions=instructions,
            input=document_text,
            model=self.model,
            output_type=MedicalDocument,
        )

        response = await self.provider.generate(request)

        if not isinstance(response.content, MedicalDocument):
            raise LLMResponseException(
                "LLM returned an unexpected response type"
            )

        return response.content
