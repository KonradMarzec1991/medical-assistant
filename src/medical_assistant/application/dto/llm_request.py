from dataclasses import dataclass

from pydantic import BaseModel


@dataclass(frozen=True)
class LLMRequest:
    instructions: str
    input: str
    model: str
    output_type: type[BaseModel] | type[str] = str
    max_output_tokens: int | None = None
    temperature: float | None = None

    def __post_init__(self):
        if not (
                self.output_type is str
                or (
                        isinstance(self.output_type, type)
                        and issubclass(self.output_type, BaseModel)
                )
        ):
            raise TypeError(
                "output_type must be str or a Pydantic BaseModel subclass"
            )

        if self.temperature is not None:
            if not 0 <= self.temperature <= 2:
                raise ValueError("temperature must be between 0 and 2")
