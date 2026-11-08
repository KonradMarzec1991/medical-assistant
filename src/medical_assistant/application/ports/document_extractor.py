from typing import Protocol


class DocumentExtractor(Protocol):
    async def extract(self, file_content: bytes, filename: str) -> str:
        ...
