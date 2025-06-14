# server.py
import os
import time
import shutil
import tempfile
import contextlib
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from src.integrations.mcp_client_manager import MCPClientManager
from src.workflows.health_advisor_graph import create_health_advisor_graph
from src.state.graph_state import HealthAdvisorState

# Load environment variables
load_dotenv()

# This object will manage the lifecycle of MCP servers.
mcp_manager = MCPClientManager(config_path="server_config.json")

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """
    The lifespan context manager for the FastAPI application.
    This function handles the setup on startup and cleanup on shutdown.
    """
    # --- Startup Logic ---
    print("Server is starting up...")
    
    # Connect to all configured MCP servers and discover their tools
    await mcp_manager.connect_to_servers()
        
    # Get API keys
    groq_api_key = os.getenv("GROQ_API_KEY")
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not groq_api_key or not google_api_key:
        raise RuntimeError("API keys for Groq and Google are required.")

    # 4. Create the LangGraph application, passing the tools to it
    app.state.health_advisor_graph = create_health_advisor_graph(
        groq_api_key=groq_api_key,
        google_api_key=google_api_key,
        mcp_tools=mcp_manager.get_tools()
    )
    print("Health Advisor Graph initialized successfully.")

    yield

    # --- Shutdown Logic ---
    await mcp_manager.cleanup()
    print("Server has shut down.")

# Initialize FastAPI with the lifespan manager
app = FastAPI(
    title="Health Advisor API",
    description="Analyzes food ingredient images using a LangGraph workflow with ReAct and MCP.",
    version="1.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "Health Advisor API is running with mounted MCP servers."}

@app.post("/analyze/")
async def analyze_food_image(file: UploadFile = File(...)):
    health_advisor_app = app.state.health_advisor_graph
    if not health_advisor_app:
        raise HTTPException(status_code=503, detail="The analysis engine is not ready.")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type.")

    temp_file_path = None
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
        print("--- Starting Analysis ---")
        start_time = time.time()

        # Run the LangGraph workflow asynchronously
        final_state = await health_advisor_app.ainvoke(initial_state)
        final_report = final_state.get("final_analysis")
        print(f"--- Analysis Completed in {(time.time() - start_time):.2f} seconds ---")

        if final_report is None:
            return JSONResponse(status_code=500, content={"error": "Analysis failed to produce a final report."})
        return final_report.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during analysis: {str(e)}")
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
