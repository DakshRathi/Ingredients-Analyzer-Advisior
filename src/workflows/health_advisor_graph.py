# src/workflows/health_advisor_graph.py
from langgraph.graph import StateGraph, START, END
import time

from src.state.graph_state import HealthAdvisorState
from src.models.data_models import CompleteHealthAnalysis # Import the final Pydantic model
from src.nodes.ingredient_extractor import create_ingredient_extractor_node
from src.nodes.analysis_node import (
    create_benefits_analysis_node,
    create_disadvantages_analysis_node,
    create_disease_analysis_node
)
from src.nodes.alternatives_recommender import create_alternatives_recommender_node

# Node names
NODE_EXTRACT_INGREDIENTS = "extract_ingredients"
NODE_ANALYZE_BENEFITS = "analyze_benefits"
NODE_ANALYZE_DISADVANTAGES = "analyze_disadvantages"
NODE_ANALYZE_DISEASES = "analyze_disease_associations"
NODE_RECOMMEND_ALTERNATIVES = "recommend_alternatives"
NODE_COMPILE_FINAL_REPORT = "compile_final_report"


def create_health_advisor_graph(groq_api_key: str, google_api_key: str, mcp_tools: list) -> StateGraph:
    """
    Creates and compiles the LangGraph for the health advisor application.
    """
    # Create node functions using their factories
    extract_ingredients_func = create_ingredient_extractor_node(groq_api_key)
    analyze_benefits_func = create_benefits_analysis_node(google_api_key, mcp_tools)
    analyze_disadvantages_func = create_disadvantages_analysis_node(google_api_key, mcp_tools)
    analyze_diseases_func = create_disease_analysis_node(google_api_key, mcp_tools)
    recommend_alternatives_func = create_alternatives_recommender_node(google_api_key, mcp_tools)

    # Define the StateGraph with the HealthAdvisorState schema
    workflow = StateGraph(HealthAdvisorState)

    # Add nodes to the graph
    workflow.add_node(NODE_EXTRACT_INGREDIENTS, extract_ingredients_func)
    workflow.add_node(NODE_ANALYZE_BENEFITS, analyze_benefits_func)
    workflow.add_node(NODE_ANALYZE_DISADVANTAGES, analyze_disadvantages_func)
    workflow.add_node(NODE_ANALYZE_DISEASES, analyze_diseases_func)
    workflow.add_node(NODE_RECOMMEND_ALTERNATIVES, recommend_alternatives_func)
    
    # Add a final node to compile results
    def compile_final_report_node(state: HealthAdvisorState) -> HealthAdvisorState:
        print("--- Compiling Final Report Node ---")
        start_time = time.time()
        
        # Create the final CompleteHealthAnalysis object
        final_report = CompleteHealthAnalysis(
            input_image_path=state["image_path"],
            extracted_data=state.get("extracted_data"),
            benefits_analysis=state.get("benefits_analysis"),
            disadvantages_analysis=state.get("disadvantages_analysis"),
            disease_analysis=state.get("disease_analysis"),
            alternatives_report=state.get("alternatives_report"),
            processing_time_seconds=state.get("total_processing_time", 0.0) # This will be set at the end of graph execution
        )

        # Generate a simple summary message for the user
        if state.get("should_stop_processing"):
            final_report.final_summary_message_for_user = (
                f"Analysis could not be completed for {state['image_path']}. "
                f"Reason: {state.get('error_message', 'Unknown error during processing.')}"
            )
        elif final_report.extracted_data:
            assessment_parts = []
            if final_report.benefits_analysis and final_report.benefits_analysis.findings:
                assessment_parts.append(f"Benefits: {', '.join(final_report.benefits_analysis.findings[:2])}.")
            if final_report.disadvantages_analysis and final_report.disadvantages_analysis.findings:
                 assessment_parts.append(f"Concerns: {', '.join(final_report.disadvantages_analysis.findings[:2])}.")
            if final_report.alternatives_report and final_report.alternatives_report.alternatives:
                assessment_parts.append(f"Consider alternatives like: {final_report.alternatives_report.alternatives[0].product_name}.")
            elif final_report.alternatives_report:
                 assessment_parts.append(final_report.alternatives_report.summary)

            final_report.final_summary_message_for_user = " ".join(assessment_parts) if assessment_parts else "Basic analysis complete."
            final_report.overall_health_assessment = final_report.final_summary_message_for_user # Simplified overall assessment

        state["final_analysis"] = final_report
        
        processing_time = time.time() - start_time
        print(f"--- Final Report Node completed in {processing_time:.2f}s ---")
        return state

    workflow.add_node(NODE_COMPILE_FINAL_REPORT, compile_final_report_node)

    # Define the workflow edges
    workflow.add_edge(START, NODE_EXTRACT_INGREDIENTS)

    # Create parallel paths after extraction
    workflow.add_edge(NODE_EXTRACT_INGREDIENTS, NODE_ANALYZE_BENEFITS)
    workflow.add_edge(NODE_EXTRACT_INGREDIENTS, NODE_ANALYZE_DISADVANTAGES)
    workflow.add_edge(NODE_EXTRACT_INGREDIENTS, NODE_ANALYZE_DISEASES)

    workflow.add_edge(NODE_ANALYZE_BENEFITS, NODE_RECOMMEND_ALTERNATIVES)
    workflow.add_edge(NODE_ANALYZE_DISADVANTAGES, NODE_RECOMMEND_ALTERNATIVES)
    workflow.add_edge(NODE_ANALYZE_DISEASES, NODE_RECOMMEND_ALTERNATIVES) 

    # Edge from alternatives recommender to the final compilation node
    workflow.add_edge(NODE_RECOMMEND_ALTERNATIVES, NODE_COMPILE_FINAL_REPORT)
    
    # Final edge to END
    workflow.add_edge(NODE_COMPILE_FINAL_REPORT, END)

    # Compile the graph
    app = workflow.compile()
    return app
