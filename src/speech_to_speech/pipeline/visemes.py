"""Timed mouth shapes shared by the pipeline and Realtime extension."""

from pydantic import BaseModel, Field, model_validator


class Viseme(BaseModel):
    """Microsoft-compatible mouth shape on the response's audio timeline."""

    viseme: int = Field(ge=0, le=21)
    start_s: float = Field(ge=0, allow_inf_nan=False)
    end_s: float = Field(ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def ordered_interval(self) -> "Viseme":
        if self.end_s < self.start_s:
            raise ValueError("end_s must be greater than or equal to start_s")
        return self
