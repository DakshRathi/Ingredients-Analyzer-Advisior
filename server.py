# server.py
import os
import time
import shutil
import tempfile
import contextlib
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Response, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from src.integrations.mcp_client_manager import MCPClientManager
from src.workflows.health_advisor_graph import create_health_advisor_graph
from src.state.graph_state import HealthAdvisorState
from src.integrations.whatsapp_service import WhatsAppService
from src.integrations.message_formatter import format_whatsapp_message


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

    app.state.whatsapp_service = WhatsAppService()
    app.state.VERIFY_TOKEN = os.getenv("META_VERIFY_TOKEN")

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
    """
    Root endpoint to check if the API is running.
    Returns a simple JSON response indicating the API status.
    """
    return {"status": "Health Advisor API is running with mounted MCP servers."}

@app.post("/analyze/")
async def analyze_food_image(file: UploadFile = File(...)):
    """
    Endpoint to analyze a food ingredient image.
    This endpoint accepts an image file, processes it through the LangGraph workflow,
    and returns a detailed health analysis report.
    """
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

@app.get("/webhook")
def verify_whatsapp_webhook(request: Request):
    """
    Handles the webhook verification challenge from Meta.
    """
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")

    if mode == "subscribe" and token == app.state.VERIFY_TOKEN:
        print("WEBHOOK_VERIFIED")
        return Response(content=challenge, status_code=200)
    else:
        raise HTTPException(status_code=403, detail="Verification token mismatch")

async def process_whatsapp_message(payload: dict):
    """
    Handles the analysis of an incoming WhatsApp image message in the background.
    """
    try:
        message_data = payload["entry"][0]["changes"][0]["value"]["messages"][0]
        from_number = message_data["from"]
        message_type = message_data["type"]

        whatsapp_service = app.state.whatsapp_service

        if message_type != "image":
            whatsapp_service.send_text_message(from_number, "Please send an image of an ingredients list to start the analysis.")
            return

        image_id = message_data["image"]["id"]
        media_url = whatsapp_service.get_media_url(image_id)
        if not media_url:
            raise ValueError("Could not get media URL from WhatsApp.")

        image_bytes = whatsapp_service.download_media(media_url)
        if not image_bytes:
            raise ValueError("Could not download image from media URL.")
        
        # Save image to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(image_bytes)
            temp_image_path = tmp.name

        try:
            health_advisor_app = app.state.health_advisor_graph
            initial_state = {"image_path": temp_image_path}
            final_state = await health_advisor_app.ainvoke(initial_state)
            final_report = final_state.get("final_analysis")
            
            if final_report:
                reply = format_whatsapp_message(final_report)
            else:
                reply = "I'm sorry, I encountered an issue analyzing your image. Please try again with a clearer picture."
            
            whatsapp_service.send_text_message(from_number, reply)
        finally:
            os.remove(temp_image_path) # Clean up the temp file

    except (KeyError, IndexError) as e:
        print(f"Error parsing webhook payload: {e}")
    except Exception as e:
        print(f"Error processing message: {e}")
        if 'from_number' in locals():
            whatsapp_service.send_text_message(from_number, "An unexpected error occurred. Please try again later.")

@app.post("/webhook")
async def receive_whatsapp_message(request: Request, background_tasks: BackgroundTasks):
    """
    Receives incoming messages from WhatsApp and triggers a background task for processing.
    """
    data = await request.json()
    background_tasks.add_task(process_whatsapp_message, data)
    return Response(status_code=200)