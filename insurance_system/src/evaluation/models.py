from pydantic import BaseModel, Field


class EvaluationResult(BaseModel):
    """Data model for evaluation results."""

    score: int = Field(description="The score of the evaluation (1 for Pass, 0 for Fail).")
    explanation: str = Field(description="Explanation of why this score was given.")
