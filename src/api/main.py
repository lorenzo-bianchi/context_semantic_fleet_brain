import asyncio
import asyncpg
import os, logging, json, uuid
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
import httpx
import re

from fastapi import Request, FastAPI, APIRouter, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import redis.asyncio as aioredis

import torch
import torch.nn.functional as F
from qdrant_client import QdrantClient, models
from pydantic import BaseModel, Field
from transformers import CLIPProcessor, CLIPModel
from google import genai
from google.genai import types

# Load Jinja templates
templates = Jinja2Templates(directory="templates")

# Load environment variables
load_dotenv()

# Standard structured logging configuration for MLOps
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- GLOBAL APP STATE ---
class AppState:
    ml_models = {}
    qdrant_client: QdrantClient = None
    pg_pool: asyncpg.Pool = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    gemini_client = None
    llm_provider = os.getenv("LLM_PROVIDER", "ollama") # "gemini" or "ollama"
    ollama_url = "http://localhost:11434/api/generate"

state = AppState()

# --- SCHEMAS ---
class LLMProviderRequest(BaseModel):
    provider: str

class CommandRequest(BaseModel):
    """Payload for sending a natural language command."""
    user_id: str = Field(..., description="User ID or calling system ID")
    instruction: str = Field(..., description="Natural language command (e.g., 'Explore the north corridor')")

class CommandResponse(BaseModel):
    status: str
    task_id: str
    message: str
    plan: list = []

class SceneRequest(BaseModel):
    image_url: str
    instruction: str

class SearchRequest(BaseModel):
    query: str
    limit: int = 3

# --- LIFESPAN ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the application lifecycle.
    Initializes the PostgreSQL connection pool, the Qdrant client, 
    Redis message broker, and loads ML/RAG models into memory.
    """
    if os.getenv("TESTING") == "true":
        logger.info("Test mode: Skipping heavy loading. API started 'empty'.")
        yield
        return

    logger.info("Starting API services: Initializing DB connections and Models...")

    # 1. Initialize Qdrant Client
    state.qdrant_client = QdrantClient(host="localhost", port=6333)

    # 1.5 Initialize Gemini Client
    if os.getenv("GEMINI_API_KEY"):
        state.gemini_client = genai.Client()
        logger.info("Gemini API Client successfully initialized.")
    else:
        logger.warning("GEMINI_API_KEY not found. The agent will not be able to reason.")

    try:
        collections = state.qdrant_client.get_collections()
        collection_names = [c.name for c in collections.collections]
        if "semantic_memory" not in collection_names:
            state.qdrant_client.create_collection(
                collection_name="semantic_memory",
                vectors_config=models.VectorParams(
                    size=512,
                    distance=models.Distance.COSINE
                )
            )
            logger.info("Qdrant collection 'semantic_memory' successfully created.")
        else:
            logger.info("Qdrant collection 'semantic_memory' already exists.")
    except Exception as e:
        logger.error(f"Error with Qdrant initialization: {e}")

    # 2. Load Multimodal Model (CLIP)
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
        db_user = os.getenv("POSTGRES_USER", "fleet_admin")
        db_password = os.getenv("POSTGRES_PASSWORD")
        db_name = os.getenv("POSTGRES_DB", "fleet_brain")
        db_host = os.getenv("POSTGRES_HOST", "127.0.0.1")

        if not db_password:
            raise ValueError("POSTGRES_PASSWORD not found. Check the .env file!")

        DB_URL = f"postgresql://{db_user}:{db_password}@{db_host}:5432/{db_name}"
        state.pg_pool = await asyncpg.create_pool(DB_URL)
        logger.info("PostgreSQL connected successfully!")

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

    # 4. Redis Connection
    logger.info("Connecting to Redis...")
    try:
        # Using decode_responses=True to handle strings automatically instead of bytes
        state.redis_client = aioredis.from_url("redis://localhost:6379", decode_responses=True)
        # Test the connection with a ping
        await state.redis_client.ping()
        logger.info("Redis connected successfully!")
    except Exception as e:
        logger.error(f"Redis connection failed: {e}")

    yield

    logger.info("Shutting down API: Cleaning up...")

    # Teardown: Close Qdrant
    if state.qdrant_client and hasattr(state.qdrant_client, "close"):
        try:
            state.qdrant_client.close()
        except Exception as e:
            logger.warning(f"Error closing Qdrant: {e}")

    # Teardown: Close Pool
    if state.pg_pool and type(state.pg_pool).__name__ != "MockPool" and hasattr(state.pg_pool, "close"):
        try:
            await state.pg_pool.close()
        except Exception as e:
            logger.warning(f"Error closing Postgres pool: {e}")

    # Teardown: Close Redis
    if state.redis_client:
        try:
            await state.redis_client.aclose()
        except Exception as e:
            logger.warning(f"Error closing Redis client: {e}")

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

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

router = APIRouter(prefix="/api/v1")

# --- UTILS ---
async def get_agent_plan(instruction: str):
    # --- 1. Object-Based Prompting (The industry standard for LLMs) ---
    prompt = f"""You are the 'Fleet Brain', the AI of a ROS 2 robot.
    Analyze the following command and extract the FULL sequence of operations.
    Allowed actions: NAVIGATE, SEARCH, PICK, DROP, COMMUNICATE.

    Command: "{instruction}"

    You MUST respond with a JSON object containing a SINGLE key called "plan". 
    The value must be the array of all actions required. Do not stop until all steps are extracted.

    Example format:
    {{
      "plan": [
        {{"action": "NAVIGATE", "target": "kitchen"}},
        {{"action": "SEARCH", "target": "bottle"}},
        {{"action": "PICK", "target": "bottle"}},
        {{"action": "COMMUNICATE", "target": "everyone about the bottle"}}
      ]
    }}

    Output strictly the JSON object. No other text."""

    raw_json = ""

    # --- 2. API Calls ---
    if state.llm_provider == "ollama":
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    state.ollama_url,
                    json={
                        "model": "gemma2:9b", 
                        "prompt": prompt, 
                        "stream": False,
                        "format": "json" 
                    },
                    timeout=60.0
                )
                response.raise_for_status()
                raw_json = response.json().get("response", "")
            except httpx.ReadTimeout:
                logger.error("Ollama timeout: Model is loading or GPU is busy.")
                return []
            except Exception as e:
                logger.error(f"Ollama connection error: {e}")
                return []
    else:
        try:
            response = state.gemini_client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )
            raw_json = response.text
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return []

    # --- 3. Simplified and Bulletproof Parsing ---
    try:
        cleaned_raw = raw_json.strip().removeprefix("```json").removesuffix("```").strip()
        parsed_data = json.loads(cleaned_raw)

        # Since we explicitly asked for {"plan": [...]}, the extraction is deterministic
        if isinstance(parsed_data, dict) and "plan" in parsed_data:
            if isinstance(parsed_data["plan"], list):
                return parsed_data["plan"]

        # Fallback if the LLM provided a top-level array despite the prompt
        if isinstance(parsed_data, list):
            return parsed_data

        logger.error(f"JSON parsed but missing 'plan' array: {parsed_data}")
        return []

    except json.JSONDecodeError:
        # Emergency regex fallback 
        try:
            match = re.search(r'\{.*\}', raw_json, re.DOTALL)
            if match:
                extracted = json.loads(match.group(0))
                if "plan" in extracted:
                    return extracted["plan"]
        except Exception as fallback_error:
            logger.error(f"RegEx fallback failed. Raw: {raw_json} | Error: {fallback_error}")

        return []

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
@app.get("/", response_class=HTMLResponse)
async def frontend(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/health", tags=["System"])
async def health_check():
    """Vital endpoint for Kubernetes Liveness and Readiness probes."""
    return {"status": "ok", "service": "fleet_brain_api"}

@app.post("/api/v1/analyze-scene")
async def analyze_scene(
    instruction: str = Form(...),
    image_file: UploadFile = File(...)
):
    """Compare a loaded image with an instruction (Multimodal)."""
    model = state.ml_models.get("clip_model")
    processor = state.ml_models.get("clip_processor")

    if not model or not processor:
        return {"status": "error", "message": "Model not loaded properly."}

    try:
        image_bytes = await image_file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        inputs = processor(text=[instruction], images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(state.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)

        image_embeds = F.normalize(outputs.image_embeds, p=2, dim=-1)
        text_embeds = F.normalize(outputs.text_embeds, p=2, dim=-1)

        cosine_similarity = torch.matmul(image_embeds, text_embeds.t())
        raw_score = cosine_similarity.item()

        return {
            "status": "success",
            "instruction": instruction,
            "filename": image_file.filename,
            "cosine_similarity_score": round(raw_score, 4),
            "match_confidence_percent": round(max(0, raw_score) * 100, 2)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

# --- ROUTER ENDPOINTS ---
@router.post("/command", response_model=CommandResponse, tags=["Orchestration"])
async def dispatch_command(payload: CommandRequest):
    """
    Receives a text command, saves it to the PostgreSQL history,
    and stores its vector embedding in Qdrant for semantic search.
    """
    logger.info(f"Received command from {payload.user_id}: {payload.instruction}")

    try:
        # 0. Generate command embedding using GPU
        embedding_vector = get_embedding(payload.instruction)
        pg_id = None

        # 1. Save command to Historical Memory (PostgreSQL)
        if state.pg_pool:
            async with state.pg_pool.acquire() as conn:
                pg_id = await conn.fetchval(
                    "INSERT INTO command_history (user_id, instruction, status) VALUES ($1, $2, $3) RETURNING id",
                    payload.user_id, payload.instruction, "pending"
                )

        # 2. Save in Qdrant 
        if state.qdrant_client:
            point_id = str(uuid.uuid4())
            state.qdrant_client.upsert(
                collection_name="semantic_memory",
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector=embedding_vector,
                        payload={
                            "pg_id": int(pg_id) if pg_id is not None else None,
                            "user_id": str(payload.user_id),
                            "instruction": str(payload.instruction),
                            "status": "pending"
                        }
                    )
                ]
            )
            logger.info(f"Command saved in Qdrant with ID: {point_id}")

        # 3. Agent Reasoning (Switchable)
        agent_plan = []
        try:
            agent_plan = await get_agent_plan(payload.instruction)
            logger.info(f"Agent plan generated via {state.llm_provider}: {agent_plan}")
        except Exception as llm_err:
            logger.error(f"Error during reasoning: {llm_err}")

        # 4. Queing task on Redis (Publisher)
        if agent_plan and state.redis_client:
            # Create a complete payload to send to the ROS 2 node
            task_payload = {
                "task_id": point_id,
                "user_id": payload.user_id,
                "instruction": payload.instruction,
                "plan": agent_plan
            }

            # Serialize in JSON and insert in 'robot_tasks_queue' 
            queue_name = "robot_tasks_queue"
            await state.redis_client.rpush(queue_name, json.dumps(task_payload))
            logger.info(f"Task {point_id} successfully queued in Redis [{queue_name}].")

    except Exception as e:
        error_details = f"{type(e).__name__}: {str(e)}"
        logger.error(f"Error processing command from {payload.user_id}: {error_details}")
        raise HTTPException(status_code=500, detail=f"Internal processing error -> {error_details}")

    return CommandResponse(
        status="accepted",
        task_id="task_mock_12345",
        message="Command accepted, analyzed by Agent and sent to ROS 2 Fleet.",
        plan=agent_plan
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
            # Convert timestamp to string for JSON serialization
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
            "vector_preview": embedding_vector[:5]
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

@router.post("/search-memory", tags=["Memory"])
async def search_memory(payload: SearchRequest):
    """Search in the semantic memory (Qdrant) the most similar commands to the entered text."""
    if not state.qdrant_client:
        raise HTTPException(status_code=500, detail="Qdrant client not connected.")

    try:
        query_vector = get_embedding(payload.query)

        search_result = state.qdrant_client.query_points(
            collection_name="semantic_memory",
            query=query_vector,
            limit=payload.limit
        ).points

        results = []
        for hit in search_result:
            results.append({
                "score": round(hit.score, 4),
                "instruction": hit.payload.get("instruction"),
                "user_id": hit.payload.get("user_id"),
                "pg_id": hit.payload.get("pg_id")
            })

        return {
            "status": "success",
            "query": payload.query,
            "matches": results
        }
    except Exception as e:
        logger.error(f"Error during semantic search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/llm-provider", tags=["System"])
async def set_llm_provider(payload: LLMProviderRequest):
    """Dynamically switch the LLM provider from the dashboard."""
    if payload.provider not in ["gemini", "ollama"]:
        raise HTTPException(status_code=400, detail="Invalid provider. Choose 'gemini' or 'ollama'.")

    state.llm_provider = payload.provider
    logger.info(f"🔄 LLM Provider switched to: {state.llm_provider}")

    return {"status": "success", "active_provider": state.llm_provider}

app.include_router(router)
