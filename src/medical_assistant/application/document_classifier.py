from enum import Enum

from pydantic import BaseModel, ConfigDict

from medical_assistant.application.dto.llm_request import LLMRequest
from medical_assistant.application.exceptions import LLMResponseException
from medical_assistant.application.ports.llm_provider import LLMProvider


class InvalidDocumentException(Exception):
    pass


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
    def __init__(self, provider: LLMProvider, model: str) -> None:
        self.provider = provider
        self.model = model

    @staticmethod
    def validate(document_text: str) -> str:
        if not isinstance(document_text, str):
            raise InvalidDocumentException(
                "Document text must be a string"
            )

        document_text = document_text.strip()

        if not document_text:
            raise InvalidDocumentException(
                "Document text cannot be empty"
            )

        if len(document_text) > 50_000:
            raise InvalidDocumentException(
                "Document text exceeds 50 000 characters"
            )

        return document_text

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

        document_text = self.validate(document_text)

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
