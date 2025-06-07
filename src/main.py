import asyncio
import os
import time
from dotenv import load_dotenv
from pathlib import Path

from src.workflows.health_advisor_graph import create_health_advisor_graph
from src.state.graph_state import HealthAdvisorState 
from src.models.data_models import ImageValidationStatus 

async def run_health_analysis(image_path: str):
    """
    Runs the health advisor workflow for a given image path.

    Args:
        image_path (str): Path to the food image to analyze.

    Returns:
        None
    """
    print(f"\n--- Starting Health Analysis for: {image_path} ---")
    start_overall_time = time.time()

    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY")

    if not groq_api_key:
        print("Error: GROQ_API_KEY not found in .env file or environment variables.")
        return

    # Create the compiled LangGraph app
    health_advisor_app = create_health_advisor_graph(groq_api_key)

    # Define the initial state to kick off the graph
    initial_state: HealthAdvisorState = {
        "image_path": image_path,
        "extracted_data": None,
        "should_stop_processing": False,
        "error_message": None,
        "benefits_analysis": None,
        "disadvantages_analysis": None,
        "disease_analysis": None,
        "alternatives_report": None,
        "final_analysis": None,
        "current_task_start_time": None,
    }

    # Invoke the graph asynchronously
    final_state = await health_advisor_app.ainvoke(initial_state)
    
    end_overall_time = time.time()
    total_processing_time = end_overall_time - start_overall_time
    
    # Access the final compiled report from the state
    final_report = final_state.get("final_analysis")

    print("\n--- Health Advisor Final Report ---")
    if final_report:
        final_report.processing_time_seconds = total_processing_time # Update with actual total time
        print(final_report.model_dump_json(indent=2))
        if final_report.final_summary_message_for_user:
            print(f"\nUser Summary:\n{final_report.final_summary_message_for_user}")

        if final_report.extracted_data and final_report.extracted_data.validation_status != ImageValidationStatus.VALID_FOOD_IMAGE:
            print(f"\nNOTICE: Initial image validation failed: {final_report.extracted_data.error_message}")
    else:
        print("Error: No final report generated. Check logs for issues.")
        print(f"Final state was: {final_state}") # For debugging if final_report is None

    print(f"--- Total processing time: {total_processing_time:.2f} seconds ---")


async def main():
    """
    Main entry point for the health advisor application.
    This function runs the health analysis on a test image.
    """

    test_image_filename = "Real.jpg"
    await run_health_analysis(test_image_filename)


if __name__ == "__main__":
    # python -m src.main
    asyncio.run(main())
