from contextlib import asynccontextmanager
from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
import logging

# Standard structured logging configuration for MLOps
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- SCHEMAS ---
class CommandRequest(BaseModel):
    """Payload for sending a natural language command."""
    user_id: str = Field(..., description="User ID or calling system ID")
    instruction: str = Field(..., description="Natural language command (e.g., 'Explore the north corridor')")

class CommandResponse(BaseModel):
    status: str
    task_id: str
    message: str

# --- LIFESPAN ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the application lifecycle.
    Here we will initialize the PostgreSQL connection pool, the Qdrant client, 
    and load LLM/RAG models into memory (if served locally) to prevent memory leaks.
    """
    logger.info("Starting API services: Initializing DB connections and Agents...")
    # TODO: Initialize Qdrant client
    # TODO: Initialize AsyncPG pool
    yield
    logger.info("Shutting down API: Closing connections and cleaning up resources...")
    # TODO: Close DB connections

# --- APP SETUP ---
app = FastAPI(
    title="Semantic Fleet Brain API",
    version="0.1.0",
    description="Agentic Orchestrator for ROS 2 robotic fleets",
    lifespan=lifespan
)

# CORS configuration (to be refined and locked down for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # TODO: Replace with specific frontend domains/IPs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

router = APIRouter(prefix="/api/v1")

# --- ENDPOINTS ---
@app.get("/", include_in_schema=False)
async def root():
    """Reindirizza la root path direttamente alla documentazione Swagger."""
    return RedirectResponse(url="/docs")

@app.get("/health", tags=["System"])
async def health_check():
    """Vital endpoint for Kubernetes Liveness and Readiness probes."""
    return {"status": "ok", "service": "fleet_brain_api"}

@router.post("/command", response_model=CommandResponse, tags=["Orchestration"])
async def dispatch_command(payload: CommandRequest):
    """
    Receives a text command, passes it to the LLM agent for decomposition 
    into physical tasks, and sends it to the ROS 2 mock queue.
    """
    logger.info(f"Received command from {payload.user_id}: {payload.instruction}")

    try:
        # Here we will insert the LangGraph/CrewAI processing logic
        # mock_task_id = llm_agent.process(payload.instruction)
        pass
    except Exception as e:
        logger.error(f"Error processing command from {payload.user_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal agent processing error.")

    return CommandResponse(
        status="accepted",
        task_id="task_mock_12345",
        message="Command accepted by the orchestrator agent."
    )

app.include_router(router)
