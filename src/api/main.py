import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
import logging

import torch
from qdrant_client import AsyncQdrantClient
from transformers import CLIPProcessor, CLIPModel

# Standard structured logging configuration for MLOps
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- GLOBAL APP STATE ---
# Using dictionaries/globals for connections and models prevents memory leaks 
# and avoids re-allocating heavy tensors on every single HTTP request.
ml_models = {}
qdrant_client = None

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
    global qdrant_client
    logger.info("Starting API services: Initializing DB connections and Models...")

    # 1. Initialize Qdrant Client (async gRPC for maximum performance)
    qdrant_client = AsyncQdrantClient(host="localhost", grpc_port=6334, prefer_grpc=True)

    # 2. Load Multimodal Model (CLIP) from local storage
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading local CLIP model on device: {device}")

    local_model_path = os.path.join(os.path.dirname(__file__), "local_models", "clip")

    if not os.path.exists(local_model_path):
        logger.error(f"Local model not found at {local_model_path}. Run 'scripts/download_model.py' first!")
    else:
        ml_models["clip_model"] = CLIPModel.from_pretrained(local_model_path).to(device)
        ml_models["clip_processor"] = CLIPProcessor.from_pretrained(local_model_path)

    # TODO: Initialize AsyncPG pool for PostgreSQL

    yield

    logger.info("Shutting down API: Closing connections and cleaning up VRAM...")

    # Teardown: Close Qdrant connection
    if qdrant_client:
        await qdrant_client.close()

    # TODO: Close AsyncPG pool

    # Teardown: Explicitly free GPU/CPU memory
    ml_models.clear()
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


import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
import logging

import torch
from qdrant_client import AsyncQdrantClient
from transformers import CLIPProcessor, CLIPModel

# Standard structured logging configuration for MLOps
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- GLOBAL APP STATE ---
# Using dictionaries/globals for connections and models prevents memory leaks 
# and avoids re-allocating heavy tensors on every single HTTP request.
ml_models = {}
qdrant_client = None

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
    global qdrant_client
    logger.info("Starting API services: Initializing DB connections and Models...")

    # 1. Initialize Qdrant Client (async gRPC for maximum performance)
    qdrant_client = AsyncQdrantClient(host="localhost", grpc_port=6334, prefer_grpc=True)

    # 2. Load Multimodal Model (CLIP) from local storage
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading local CLIP model on device: {device}")

    local_model_path = os.path.join(os.path.dirname(__file__), "local_models", "clip")

    if not os.path.exists(local_model_path):
        logger.error(f"Local model not found at {local_model_path}. Run 'scripts/download_model.py' first!")
    else:
        ml_models["clip_model"] = CLIPModel.from_pretrained(local_model_path).to(device)
        ml_models["clip_processor"] = CLIPProcessor.from_pretrained(local_model_path)

    # TODO: Initialize AsyncPG pool for PostgreSQL
    
    yield
    
    logger.info("Shutting down API: Closing connections and cleaning up VRAM...")
    
    # Teardown: Close Qdrant connection
    if qdrant_client:
        await qdrant_client.close()
        
    # TODO: Close AsyncPG pool
    
    # Teardown: Explicitly free GPU/CPU memory
    ml_models.clear()
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

@router.get("/test-embedding", tags=["System Test"])
async def test_embedding(text: str = "Test di generazione embedding"):
    """
    Endpoint temporaneo per verificare che il modello CLIP sia caricato
    correttamente in memoria e riesca a vettorizzare il testo.
    """
    processor = ml_models.get("clip_processor")
    model = ml_models.get("clip_model")

    if not processor or not model:
        raise HTTPException(status_code=500, detail="Modelli non caricati in memoria.")

    try:
        # 1. Pre-processamento (Tokenizzazione)
        inputs = processor(text=text, return_tensors="pt", padding=True, truncation=True)

        # 2. Spostamento dei tensori sul device corretto (CPU o CUDA)
        device = model.device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 3. Inferenza (Senza calcolare i gradienti per risparmiare memoria)
        with torch.no_grad():
            outputs = model.get_text_features(**inputs)

        # 4. Estrazione della lista di float
        embedding_vector = outputs[0].tolist()

        return {
            "status": "success",
            "input_text": text,
            "vector_dimension": len(embedding_vector),
            "vector_preview": embedding_vector[:5] # Mostriamo solo i primi 5 numeri
        }
    except Exception as e:
        logger.error(f"Errore durante la generazione dell'embedding: {e}")
        raise HTTPException(status_code=500, detail="Errore interno di inferenza.")

app.include_router(router)app.include_router(router)
