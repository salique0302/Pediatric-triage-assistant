from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import List, Optional, Literal


class ProductRef(BaseModel):
    """Reference to a Mumzworld product"""
    id: str
    name: str


class TriageRequest(BaseModel):
    """Input schema for triage request"""
    model_config = ConfigDict(strict=True)

    symptoms: str = Field(
        ...,
        min_length=10,
        max_length=1000,
        description="Free-text description of child's symptoms"
    )
    child_age_months: int = Field(
        ...,
        ge=0,
        le=216,
        description="Child's age in months (0-216 = 0-18 years)"
    )
    temperature_celsius: Optional[float] = Field(
        None,
        ge=35.0,
        le=43.0,
        description="Body temperature in Celsius if measured"
    )
    duration_hours: Optional[int] = Field(
        None,
        ge=0,
        description="Duration of symptoms in hours"
    )
    additional_context: Optional[str] = Field(
        None,
        max_length=500,
        description="Any additional context or information"
    )


class TriageResponse(BaseModel):
    """Output schema for triage response"""
    model_config = ConfigDict(strict=True)

    input_language: Literal["en", "ar"]
    child_age_months: int
    symptoms_understood: List[str] = Field(
        ...,
        description="List of symptoms extracted from input"
    )
    severity: Literal["low", "medium", "high", "emergency"]
    severity_reasoning: str = Field(
        ...,
        min_length=10,
        description="Explanation for the severity assessment"
    )
    home_care: List[str] = Field(
        ...,
        description="Concrete home care steps the parent can take"
    )
    suggested_action: str = Field(
        ...,
        min_length=10,
        description="Clear, actionable next step"
    )
    defer_to_doctor: bool
    defer_reason: Optional[str] = Field(
        None,
        description="Required if defer_to_doctor is True"
    )
    relevant_products: List[ProductRef] = Field(
        default_factory=list,
        description="Relevant Mumzworld products if applicable"
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in the assessment (0.0-1.0)"
    )
    out_of_scope: bool = Field(
        default=False,
        description="True if query is outside pediatric symptom triage scope"
    )
    disclaimer: str = Field(
        ...,
        min_length=20,
        description="Medical disclaimer - always present"
    )
    retrieved_context_used: bool = Field(
        default=True,
        description="Whether relevant context was retrieved from knowledge base"
    )

    @field_validator('defer_reason')
    @classmethod
    def validate_defer_reason(cls, v, info):
        """Ensure defer_reason is provided when defer_to_doctor is True"""
        if info.data.get('defer_to_doctor') and v is None:
            raise ValueError('defer_reason is required when defer_to_doctor is True')
        return v
