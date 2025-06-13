# server.py

import os
import shutil
import tempfile
import contextlib
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

# Import the mcp_app objects from your server files
from src.mcp_servers.serpapi_server import mcp as serpapi_mcp_app
from src.mcp_servers.website_content_server import mcp as scraper_mcp_app

# Import your application components
from src.tools.mcp_tools import ALL_MCP_LANGCHAIN_TOOLS
from src.workflows.health_advisor_graph import create_health_advisor_graph
from src.state.graph_state import HealthAdvisorState

load_dotenv()

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the application's startup and shutdown events.
    """
    # --- Startup Logic ---
    print("Server is starting up...")
    
    # 1. Start the session managers for our MCP applications. This is required by fastmcp.
    async with contextlib.AsyncExitStack() as stack:
        await stack.enter_async_context(serpapi_mcp_app.session_manager.run())
        await stack.enter_async_context(scraper_mcp_app.session_manager.run())
        
        # 2. Get API keys
        groq_api_key = os.getenv("GROQ_API_KEY")
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not groq_api_key or not google_api_key:
            raise RuntimeError("API keys for Groq and Google are required.")
            
        # 3. Discover all MCP tools available in memory    
        all_mcp_tools = ALL_MCP_LANGCHAIN_TOOLS

        
        print(f"Discovered in-memory MCP tools: {[t.name for t in all_mcp_tools]}")

        # 4. Create the LangGraph application, passing the tools to it
        app.state.health_advisor_graph = create_health_advisor_graph(
            groq_api_key=groq_api_key,
            google_api_key=google_api_key,
            mcp_tools=all_mcp_tools
        )
        print("Health Advisor Graph initialized successfully.")

        yield # The application is now running

    # --- Shutdown Logic ---
    print("Server has shut down.")

# Initialize FastAPI with the lifespan manager
app = FastAPI(
    title="Health Advisor API (Streamable HTTP)",
    version="2.0.0",
    lifespan=lifespan
)

# Mount the MCP servers as sub-applications, making them accessible over HTTP
# This makes them standard and reusable for other potential clients in the future.
app.mount("/serpapi", serpapi_mcp_app.streamable_http_app())
app.mount("/scraper", scraper_mcp_app.streamable_http_app())

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
    # ... (This endpoint logic remains exactly the same as before) ...
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
        print(f"Starting analysis for temporary file: {temp_file_path}")
        # Run the LangGraph workflow asynchronously
        final_state = await health_advisor_app.ainvoke(initial_state)
        final_report = final_state.get("final_analysis")

        if final_report is None:
            return JSONResponse(status_code=500, content={"error": "Analysis failed to produce a final report."})
        return final_report.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during analysis: {str(e)}")
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
