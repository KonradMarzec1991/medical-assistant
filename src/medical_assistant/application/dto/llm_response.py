from dataclasses import dataclass


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int
    output_tokens: int
    max_output_tokens: int | None = None

    @property
    def total_tokens(self) -> int | None:
        return self.input_tokens + self.output_tokens
