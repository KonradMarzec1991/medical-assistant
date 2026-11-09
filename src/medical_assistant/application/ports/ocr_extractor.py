from typing import Protocol


class OCRExtractor(Protocol):
    async def extract(self, file_content: bytes) -> str:
        ...
