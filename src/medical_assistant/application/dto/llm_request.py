from dataclasses import dataclass


@dataclass(frozen=True)
class LLMRequest:
    instructions: str
    input: str
    model: str
    max_output_tokens: int | None = None
    temperature: float | None = None

    def __post_init__(self):
        if self.temperature is not None:
            if not 0 <= self.temperature <= 2:
                raise ValueError("temperature must be between 0 and 2")
