import os
import logging
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from qdrant_client import AsyncQdrantClient
from transformers import CLIPProcessor, CLIPModel

import asyncpg
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Standard structured logging configuration for MLOps
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- GLOBAL APP STATE ---
class AppState:
    ml_models = {}
    qdrant_client: AsyncQdrantClient = None
    pg_pool: asyncpg.Pool = None
    device: str = "cpu"

state = AppState()

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
    Initializes the PostgreSQL connection pool, the Qdrant client, 
    and loads ML/RAG models into memory to prevent memory leaks.
    """
    logger.info("Starting API services: Initializing DB connections and Models...")

    # 1. Initialize Qdrant Client (async gRPC for maximum performance)
    state.qdrant_client = AsyncQdrantClient(host="localhost", grpc_port=6334, prefer_grpc=True)

    # 2. Load Multimodal Model (CLIP) from local storage
    state.device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading local CLIP model on device: {state.device}")
    local_model_path = os.path.join(os.path.dirname(__file__), "local_models", "clip")
    if not os.path.exists(local_model_path):
        logger.error(f"Local model not found at {local_model_path}. Run 'scripts/download_model.py' first!")
    else:
        state.ml_models["clip_model"] = CLIPModel.from_pretrained(local_model_path).to(state.device)
        state.ml_models["clip_processor"] = CLIPProcessor.from_pretrained(local_model_path)

    # 3. PostgreSQL
    logger.info("Connecting to PostgreSQL...")
    try:
        # Read all parameters from the .env file (with fallback to defaults if missing)
        db_user = os.getenv("POSTGRES_USER", "fleet_admin")
        db_password = os.getenv("POSTGRES_PASSWORD")
        db_name = os.getenv("POSTGRES_DB", "fleet_brain")
        db_host = os.getenv("POSTGRES_HOST", "127.0.0.1")

        if not db_password:
            raise ValueError("POSTGRES_PASSWORD not found. Check the .env file!")

        # Dynamic and clean connection URL
        DB_URL = f"postgresql://{db_user}:{db_password}@{db_host}:5432/{db_name}"

        state.pg_pool = await asyncpg.create_pool(DB_URL)
        logger.info("PostgreSQL connected successfully!")

        # --- Automatic creation of the history table ---
        async with state.pg_pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS command_history (
                    id SERIAL PRIMARY KEY,
                    user_id VARCHAR(50) NOT NULL,
                    instruction TEXT NOT NULL,
                    status VARCHAR(20) NOT NULL,
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
            """)
            logger.info("Database tables verified.")

    except Exception as e:
        logger.exception("Database connection failed")

    yield

    logger.info("Shutting down API: Closing connections and cleaning up VRAM...")

    # Teardown: Close Qdrant connection
    if state.qdrant_client:
        await state.qdrant_client.close()

    # Teardown: Close AsyncPG pool
    if state.pg_pool:
        await state.pg_pool.close()

    # Teardown: Explicitly free GPU/CPU memory
    state.ml_models.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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

# --- UTILS ---
def get_embedding(text: str):
    """Reliably extract the pure 512-dim text embedding."""
    model = state.ml_models.get("clip_model")
    processor = state.ml_models.get("clip_processor")

    if not model or not processor:
        raise RuntimeError("Model or processor not loaded.")

    inputs = processor(text=text, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(state.device) for k, v in inputs.items() if v is not None}

    with torch.no_grad():
        output = model.text_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )
        return output.pooler_output.detach().cpu().reshape(-1).tolist()

# --- ENDPOINTS ---
@app.get("/", include_in_schema=False)
async def root():
    """Redirects the root path directly to the Swagger documentation."""
    return RedirectResponse(url="/docs")

@app.get("/health", tags=["System"])
async def health_check():
    """Vital endpoint for Kubernetes Liveness and Readiness probes."""
    return {"status": "ok", "service": "fleet_brain_api"}

@router.post("/command", response_model=CommandResponse, tags=["Orchestration"])
async def dispatch_command(payload: CommandRequest):
    """
    Receives a text command, saves it to the PostgreSQL history,
    and (future) passes it to the LLM agent for decomposition.
    """
    logger.info(f"Received command from {payload.user_id}: {payload.instruction}")

    try:
        # 1. Save command to Historical Memory (PostgreSQL)
        async with state.pg_pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO command_history (user_id, instruction, status) VALUES ($1, $2, $3)",
                payload.user_id, payload.instruction, "pending"
            )

        # 2. Here we will insert the LangGraph/CrewAI processing logic and Qdrant storage
        pass
    except Exception as e:
        logger.error(f"Error processing command from {payload.user_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal agent processing error.")

    return CommandResponse(
        status="accepted",
        task_id="task_mock_12345",
        message="Command accepted and saved to history."
    )

@router.get("/history", tags=["Orchestration"])
async def get_command_history(limit: int = 10):
    """Retrieves the most recent commands sent to the fleet."""
    if not state.pg_pool:
        raise HTTPException(status_code=500, detail="Database not connected.")

    try:
        async with state.pg_pool.acquire() as conn:
            # Retrieve records in descending order (newest first)
            records = await conn.fetch(
                "SELECT id, user_id, instruction, status, timestamp FROM command_history ORDER BY timestamp DESC LIMIT $1",
                limit
            )
            return [dict(record) for record in records]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/test-embedding", tags=["System Test"])
async def test_embedding_endpoint(text: str = "Embedding generation test"):
    """
    Temporary endpoint to verify that the CLIP model is loaded
    correctly in memory and can vectorize text.
    """
    try:
        # Use the centralized utility function for the embedding
        embedding_vector = get_embedding(text)

        return {
            "status": "success",
            "input_text": text,
            "vector_dimension": len(embedding_vector),
            "vector_preview": embedding_vector[:5] # Show only the first 5 numbers
        }
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Error during embedding generation: {e}")
        raise HTTPException(status_code=500, detail="Internal inference error.")

@router.get("/test-db", tags=["System Test"])
async def test_db():
    """Verifies that PostgreSQL responds correctly."""
    if not state.pg_pool:
        raise HTTPException(status_code=500, detail="Database not connected.")

    try:
        # Get a connection from the pool and run a simple query
        async with state.pg_pool.acquire() as conn:
            version = await conn.fetchval('SELECT version();')
            return {"status": "success", "db_version": version}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

app.include_router(router)
