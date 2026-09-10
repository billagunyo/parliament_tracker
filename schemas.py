from pydantic import BaseModel, Field


class BillAnalysisSchema(BaseModel):
    summary: str = Field(
        description="A concise 2-sentence summary of what the bill proposes."
    )
    economic_impact: int = Field(
        description="Score from -2 (High Economic Burden) to +2 (High Economic Benefit).",
        ge=-2,
        le=2,
    )
    social_impact: int = Field(
        description="Score from -2 (Restricts Rights/Services) to +2 (Protects Rights/Services).",
        ge=-2,
        le=2,
    )
    overall_impact_score: int = Field(
        description="Final net score ranging from -2 (Very Negative) to +2 (Very Positive).",
        ge=-2,
        le=2,
    )
    impact_justification: str = Field(
        description="Brief justification for why these scores were assigned."
    )