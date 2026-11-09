from fastapi import APIRouter, Depends, File, UploadFile, HTTPException

from medical_assistant.api.dependencies import get_document_processing_service
from medical_assistant.application.document_processing_service import DocumentProcessingService
from medical_assistant.infrastructure.document_processing.pymupdf_extractor import ScannedDocumentException, \
    DocumentExtractionException

router = APIRouter()

MAX_FILE_SIZE = 10 * 1024 * 1024


@router.post("/process")
async def process_document(
    file: UploadFile = File(...),
    service: DocumentProcessingService = Depends(get_document_processing_service),
):
    file_content = await file.read(MAX_FILE_SIZE + 1)
    if not file_content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    if len(file_content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File exceeds maximum size of 10 MB")

    if not file_content.startswith(b"%PDF-") or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=415,
            detail="Uploaded file is not PDF",
        )

    filename = file.filename

    try:
        content = await service.process(file_content, filename)
    except ScannedDocumentException as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except DocumentExtractionException as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    return content