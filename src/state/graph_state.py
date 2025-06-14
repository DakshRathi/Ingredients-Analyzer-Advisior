# src/state/graph_state.py
from typing import Optional, Annotated
from typing_extensions import TypedDict
from langgraph.channels.last_value import LastValue

from src.models.data_models import (
    ExtractedIngredientsData,
    HealthAnalysisReport,
    HealthyAlternativesReport,
    CompleteHealthAnalysis
)

class HealthAdvisorState(TypedDict, total=False):
    image_path: Annotated[str, LastValue(str)]
    extracted_data: Annotated[Optional[ExtractedIngredientsData], LastValue(ExtractedIngredientsData)]
    should_stop_processing: Annotated[bool, LastValue(bool)]
    error_message: Annotated[Optional[str], LastValue(str)]

    # Analysis results
    benefits_analysis: Annotated[Optional[HealthAnalysisReport], LastValue(HealthAnalysisReport)]
    disadvantages_analysis: Annotated[Optional[HealthAnalysisReport], LastValue(HealthAnalysisReport)]
    disease_analysis: Annotated[Optional[HealthAnalysisReport], LastValue(HealthAnalysisReport)]
    alternatives_report: Annotated[Optional[HealthyAlternativesReport], LastValue(HealthyAlternativesReport)]

    final_analysis: Annotated[Optional[CompleteHealthAnalysis], LastValue(CompleteHealthAnalysis)]