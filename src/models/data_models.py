from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Optional
from enum import Enum

class ImageValidationStatus(str, Enum):
    VALID_FOOD_IMAGE = "valid_food_image"
    INVALID_NOT_FOOD = "invalid_not_food"
    INVALID_NO_INGREDIENTS = "invalid_no_ingredients"
    INVALID_POOR_QUALITY = "invalid_poor_quality"
    ERROR = "error"

class NutritionalInfo(BaseModel):
    calories_per_100g: Optional[float] = Field(None, ge=0, description="Calories per 100 grams")
    protein_grams: Optional[float] = Field(None, ge=0, description="Protein in grams per 100g")
    fat_grams: Optional[float] = Field(None, ge=0, description="Fat in grams per 100g")
    carbohydrates_grams: Optional[float] = Field(None, ge=0, description="Carbohydrates in grams per 100g")
    fiber_grams: Optional[float] = Field(0, ge=0, description="Fiber in grams per 100g") # Default to 0 if not specified
    sugar_grams: Optional[float] = Field(None, ge=0, description="Sugar in grams per 100g")
    sodium_mg: Optional[float] = Field(None, ge=0, description="Sodium in mg per 100g")

class ExtractedIngredientsData(BaseModel):
    validation_status: ImageValidationStatus
    is_food_product: Optional[bool] = Field(None, description="Whether the image contains a food product")
    has_ingredients_list: Optional[bool] = Field(None, description="Whether a visible ingredients list was detected")
    image_quality_assessment: Optional[str] = Field(None, description="e.g., 'good', 'fair', 'poor'")
    ingredients: List[str] = Field(default_factory=list, description="List of ingredients")
    allergens: List[str] = Field(default_factory=list, description="List of allergens mentioned")
    warnings: List[str] = Field(default_factory=list, description="Dietary warnings or notices")
    nutritional_info: Optional[NutritionalInfo] = Field(None, description="Nutritional information if available")
    product_name: Optional[str] = Field(None, description="Name of the food product")
    brand: Optional[str] = Field(None, description="Brand name if visible")
    confidence_score: float = Field(default=0.5, ge=0, le=1, description="Confidence in extraction accuracy")
    error_message: Optional[str] = Field(None, description="Error message if extraction or validation failed")

    @field_validator('ingredients', mode='before')
    @classmethod
    def clean_ingredients_list(cls, v):
        if isinstance(v, list):
            return [str(item).strip() for item in v if str(item).strip()]
        return []

    @field_validator('allergens', 'warnings', mode='before')
    @classmethod
    def clean_string_list(cls, v):
        if isinstance(v, list):
            return [str(item).strip() for item in v if str(item).strip()]
        return []
    
    # Ensure that error_message is present if status is ERROR
    @model_validator(mode='after')
    def check_error_status(cls, values):
        status = values.validation_status
        error_msg = values.error_message
        if status == ImageValidationStatus.ERROR and not error_msg:
            values.error_message = "An unspecified error occurred during image processing."
        return values

class HealthAnalysisReport(BaseModel):
    analysis_type: str = Field(description="Type of analysis (benefits/disadvantages/disease_associations)")
    findings: List[str] = Field(description="List of key findings")
    detailed_analysis: str = Field(description="Detailed analysis text")
    confidence_level: str = Field(description="High/Medium/Low confidence in the analysis")
    sources_consulted: List[str] = Field(default_factory=list, description="Web sources or knowledge areas used")
    health_score_impact: Optional[float] = Field(None, ge=-10, le=10, description="Impact on overall health score (-10 to +10)")

class HealthyAlternative(BaseModel):
    product_name: str = Field(description="Name of alternative product/ingredient")
    reason: str = Field(description="Why this is a healthier choice")
    availability: Optional[str] = Field(None, description="General availability (e.g., 'common supermarkets')")
    nutritional_comparison: Optional[str] = Field(None, description="Brief comparison of key nutritional aspects")

class HealthyAlternativesReport(BaseModel):
    alternatives: List[HealthyAlternative] = Field(default_factory=list, description="List of recommended alternatives")
    summary: str = Field(description="Overall summary of why alternatives are suggested or if the product is okay")

    @field_validator('alternatives')
    @classmethod
    def validate_alternatives_count(cls, v):
        if len(v) > 3: # Allow 0-3 alternatives
            raise ValueError("Maximum 3 alternatives should be provided")
        return v

class CompleteHealthAnalysis(BaseModel):
    input_image_path: str
    extracted_data: ExtractedIngredientsData
    benefits_analysis: Optional[HealthAnalysisReport] = None
    disadvantages_analysis: Optional[HealthAnalysisReport] = None
    disease_analysis: Optional[HealthAnalysisReport] = None
    alternatives_report: Optional[HealthyAlternativesReport] = None
    overall_health_assessment: Optional[str] = Field(None, description="A brief textual summary of the product's healthiness")
    processing_time_seconds: Optional[float] = None
    final_summary_message_for_user: Optional[str] = None
