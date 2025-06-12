# server.py
import os
import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from src.workflows.health_advisor_graph import create_health_advisor_graph
from src.state.graph_state import HealthAdvisorState

# Load environment variables from .env file
load_dotenv()

# This global variable will hold the compiled LangGraph application
health_advisor_app = None

async def startup_event():
    """
    This function runs when the server starts.
    It initializes the LangGraph application, making it ready to handle requests.
    """
    global health_advisor_app
    print("Server is starting up...")
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise RuntimeError("GROQ_API_KEY not set in environment variables. Please check your .env file.")
    
    # Create and compile the graph
    health_advisor_app = create_health_advisor_graph(groq_api_key)
    print("Health Advisor Graph initialized successfully.")

async def lifespan(app: FastAPI):
    await startup_event()
    yield
    print("Shutting down...")

# Initialize FastAPI app
app = FastAPI(
    title="Health Advisor API",
    description="Analyzes food ingredient images using a LangGraph workflow.",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow Cross-Origin Resource Sharing (CORS) for the Streamlit app
# This is crucial for allowing the frontend to communicate with the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    """A simple endpoint to check if the server is running."""
    return {"status": "Health Advisor API is running"}

@app.post("/analyze/")
async def analyze_food_image(file: UploadFile = File(...)):
    """
    The main endpoint that receives an image, processes it, and returns the analysis.
    """
    if not health_advisor_app:
        raise HTTPException(status_code=503, detail="The analysis engine is not ready. Please try again later.")

    # Validate that the uploaded file is an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"Invalid file type: {file.content_type}. Please upload an image.")

    # Securely save the uploaded file to a temporary location
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_file_path = tmp.name
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save the uploaded file: {e}")

    # Prepare the initial state for the LangGraph workflow
    initial_state: HealthAdvisorState = {
        "image_path": temp_file_path,
        "extracted_data": None,
        "should_stop_processing": False,
        "error_message": None,
        "benefits_analysis": None,
        "disadvantages_analysis": None,
        "disease_analysis": None,
        "alternatives_report": None,
        "final_analysis": None,
    }

    try:
        print(f"Starting analysis for temporary file: {temp_file_path}")
        # Run the LangGraph workflow asynchronously
        final_state = await health_advisor_app.ainvoke(initial_state)
        final_report = final_state.get("final_analysis")

        if final_report is None:
            return JSONResponse(status_code=500, content={"error": "Analysis failed to produce a final report."})

        # Return the final report's data as a JSON object
        return final_report.model_dump()

    except Exception as e:
        # Catch any unexpected errors during the graph execution
        raise HTTPException(status_code=500, detail=f"An error occurred during analysis: {e}")
    finally:
        # Crucially, clean up the temporary file after processing is complete
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            print(f"Cleaned up temporary file: {temp_file_path}")

