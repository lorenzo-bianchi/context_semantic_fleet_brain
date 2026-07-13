import asyncio
import base64
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Any

import asyncpg
import redis.asyncio as aioredis
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from fastapi import (
    APIRouter,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from google import genai
from PIL import Image
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient, models
from transformers import CLIPModel, CLIPProcessor

from agent.builder import build_agent_graph

# Initialize graph
agent_app = build_agent_graph()

# Load Jinja templates
templates = Jinja2Templates(directory="templates")

# Load environment variables
load_dotenv()

# Standard structured logging configuration for MLOps
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- GLOBAL APP STATE ---
class AppState:
    ml_models: dict[str, Any] = {}
    qdrant_client: Any = None
    pg_pool: Any = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    gemini_client: Any = None
    llm_provider: str = os.getenv("LLM_PROVIDER", "ollama")  # "gemini" or "ollama"
    ollama_url: str = "http://localhost:11434/api/generate"
    redis_client: Any = None
    memory_worker_task: Any = None
    feedback_worker_task: asyncio.Task | None = None


state = AppState()


# --- SCHEMAS ---
class LLMProviderRequest(BaseModel):
    provider: str


class CommandRequest(BaseModel):
    """Payload for sending a natural language command."""

    user_id: str = Field(..., description="User ID or calling system ID")
    instruction: str = Field(
        ..., description="Natural language command (e.g., 'Explore the north corridor')"
    )


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
    try:
        logger.info("🧹 Resetting Qdrant collection 'semantic_memory'...")
        state.qdrant_client.recreate_collection(
            collection_name="semantic_memory",
            vectors_config=models.VectorParams(size=512, distance=models.Distance.COSINE),
        )
        logger.info("Qdrant collection reset successfully.")
    except Exception as e:
        logger.error(f"Error resetting Qdrant: {e}")

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
                vectors_config=models.VectorParams(size=512, distance=models.Distance.COSINE),
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
        logger.error(
            f"Local model not found at {local_model_path}. Run 'scripts/download_model.py' first!"
        )
    else:
        state.ml_models["clip_model"] = CLIPModel.from_pretrained(local_model_path).to(state.device)  # type: ignore
        state.ml_models["clip_processor"] = CLIPProcessor.from_pretrained(local_model_path)  # type: ignore

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

    except Exception:
        logger.exception("Database connection failed")

    # 4. Redis Connection
    logger.info("Connecting to Redis...")
    try:
        # Using decode_responses=True to handle strings automatically instead of bytes
        state.redis_client = aioredis.from_url("redis://localhost:6379", decode_responses=True)
        # Test the connection with a ping
        await state.redis_client.ping()
        logger.info("Redis connected successfully!")

        await state.redis_client.delete("semantic_memory_queue")
        logger.info("🧹 Semantic memory queue cleared at startup.")
    except Exception as e:
        logger.error(f"Redis connection failed: {e}")

    # 5. Start Background Workers
    state.memory_worker_task = asyncio.create_task(semantic_memory_worker())
    state.feedback_worker_task = asyncio.create_task(task_feedback_worker())

    yield

    logger.info("Shutting down API: Cleaning up...")

    if hasattr(state, "memory_worker_task"):
        state.memory_worker_task.cancel()

    if hasattr(state, "feedback_worker_task"):
        state.feedback_worker_task.cancel()

    # Teardown: Close Qdrant
    if state.qdrant_client and hasattr(state.qdrant_client, "close"):
        try:
            state.qdrant_client.close()
        except Exception as e:
            logger.warning(f"Error closing Qdrant: {e}")

    # Teardown: Close Pool
    if (
        state.pg_pool
        and type(state.pg_pool).__name__ != "MockPool"
        and hasattr(state.pg_pool, "close")
    ):
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
    lifespan=lifespan,
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
def get_embedding(text: str):
    """Reliably extract the pure 512-dim text embedding."""
    model = state.ml_models.get("clip_model")
    processor = state.ml_models.get("clip_processor")

    if not model or not processor:
        raise RuntimeError("Model or processor not loaded.")

    inputs = processor(text=text, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(state.device) for k, v in inputs.items() if v is not None}

    with torch.no_grad():
        outputs = model.text_model(**inputs)
        pooled_output = outputs.pooler_output
        text_embeds = model.text_projection(pooled_output)
        return F.normalize(text_embeds, p=2, dim=-1).squeeze().tolist()


async def semantic_memory_worker():
    """Loops to listen to the Redis queue, vectorizes images, and saves them to Qdrant."""
    logger.info("🧠 Semantic Memory Worker started in background.")

    # Create the folder to save the physical images (if it doesn't exist)
    img_folder = os.path.join(os.path.dirname(__file__), "memory", "images")
    os.makedirs(img_folder, exist_ok=True)

    while True:
        try:
            # 1. Pop the payload from Redis
            result = await state.redis_client.blpop("semantic_memory_queue", timeout=1)

            if not result:
                continue

            _, task_data = result
            data = json.loads(task_data)

            if "image" not in data:
                logger.warning("Received payload without image. Ignoring.")
                continue

            # 2. Extract data from the payload
            x, y, z, yaw = data["x"], data["y"], data["z"], data["yaw"]
            b64_image = data["image"]
            timestamp = data["timestamp"]

            # 3. Base64 decoding and saving to disk
            img_data = base64.b64decode(b64_image)
            image = Image.open(BytesIO(img_data)).convert("RGB")

            img_filename = f"discovery_{int(timestamp)}.jpg"
            img_path = os.path.join(img_folder, img_filename)
            image.save(img_path)

            # 4. CLIP Inference: Turning the image into a Vector
            processor = state.ml_models["clip_processor"]
            model = state.ml_models["clip_model"]

            inputs = processor(images=image, return_tensors="pt")
            inputs = {k: v.to(state.device) for k, v in inputs.items()}

            with torch.no_grad():
                vision_outputs = model.vision_model(**inputs)
                raw_tensor = vision_outputs.pooler_output
                projected_tensor = model.visual_projection(raw_tensor)
                embedding_vector = F.normalize(projected_tensor, p=2, dim=-1).squeeze().tolist()

            # 5. Save to Qdrant
            point_id = str(uuid.uuid4())
            state.qdrant_client.upsert(
                collection_name="semantic_memory",
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector=embedding_vector,
                        payload={
                            "type": "visual_discovery",
                            "x": x,
                            "y": y,
                            "z": z,
                            "yaw": yaw,
                            "image_path": img_path,
                            "timestamp": timestamp,
                        },
                    )
                ],
            )
            logger.info(
                f"🌟 New discovery stored! Object vectorized at coordinates: ({x}, {y}, {z})"
            )

        except asyncio.CancelledError:
            logger.info("Semantic Memory Worker shutting down.")
            break
        except Exception as e:
            logger.error(f"Error processing visual memory: {type(e).__name__} - {e}")
            await asyncio.sleep(2)


async def task_feedback_worker():
    """Listens for drone failures and replans the graph, discarding incorrect points."""
    logger.info("🔄 Task Feedback Worker started in background.")
    from agent.builder import build_agent_graph

    feedback_agent = build_agent_graph()

    while True:
        try:
            result = await state.redis_client.blpop("task_feedback_queue", timeout=1)
            if not result:
                continue

            _, data_str = result
            feedback = json.loads(data_str)
            failed_point_id = feedback["failed_point_id"]
            instruction = feedback["instruction"]
            user_id = feedback["user_id"]

            logger.info(f"⚠️ Drone reported failure at point {failed_point_id}. Re-planning...")

            # Re-run the graph adding the failed point to the blacklist
            initial_state = {
                "instruction": instruction,
                "inspected_point_ids": [failed_point_id],
                "current_telemetry": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
            }

            final_state = await feedback_agent.ainvoke(initial_state)

            ros2_plan = [
                step.model_dump(exclude_none=True) for step in final_state.get("final_plan", [])
            ]

            task_payload = {
                "task_id": str(uuid.uuid4()),
                "user_id": user_id,
                "instruction": instruction,
                "plan": ros2_plan,
            }

            await state.redis_client.rpush("robot_tasks_queue", json.dumps(task_payload))
            logger.info("🚀 New fallback plan queued!")

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in feedback worker: {e}")
            await asyncio.sleep(2)


# --- ENDPOINTS ---
@app.get("/", response_class=HTMLResponse)
async def frontend(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/health", tags=["System"])
async def health_check():
    """Vital endpoint for Kubernetes Liveness and Readiness probes."""
    return {"status": "ok", "service": "fleet_brain_api"}


@app.post("/api/v1/analyze-scene")
async def analyze_scene(instruction: str = Form(...), image_file: UploadFile = File(...)):
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
            "match_confidence_percent": round(max(0, raw_score) * 100, 2),
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.websocket("/ws/video_stream")
async def video_stream(websocket: WebSocket):
    await websocket.accept()
    pubsub = state.redis_client.pubsub()

    await pubsub.subscribe("live_video_stream")

    try:
        while True:
            message = await pubsub.get_message(ignore_subscribe_messages=True)

            if message and message["type"] == "message":
                data = message["data"]
                if isinstance(data, bytes):
                    data = data.decode("utf-8")

                await websocket.send_text(data)

            await asyncio.sleep(0.01)

    except WebSocketDisconnect:
        logger.info("📱 Frontend disconnected from video stream.")
        await pubsub.unsubscribe("live_video_stream")


@app.websocket("/ws/terminal_logs")
async def terminal_logs_stream(websocket: WebSocket):
    await websocket.accept()
    pubsub = state.redis_client.pubsub()

    await pubsub.subscribe("terminal_logs")

    try:
        while True:
            message = await pubsub.get_message(ignore_subscribe_messages=True)

            if message and message["type"] == "message":
                data = message["data"]
                if isinstance(data, bytes):
                    data = data.decode("utf-8")

                await websocket.send_text(data)

            await asyncio.sleep(0.01)

    except WebSocketDisconnect:
        logger.info("💻 Frontend disconnected from terminal logs stream.")
        await pubsub.unsubscribe("terminal_logs")


# --- ROUTER ENDPOINTS ---
@router.post("/command", response_model=CommandResponse, tags=["Orchestration"])
async def dispatch_command(payload: CommandRequest):
    """
    Receives a text command, saves it to the PostgreSQL history,
    and passes it to the LangGraph agent for semantic and spatial reasoning.
    """
    logger.info(f"Received command from {payload.user_id}: {payload.instruction}")

    try:
        # 1. Save command to Historical Memory (PostgreSQL)
        point_id = str(uuid.uuid4())

        if state.pg_pool:
            async with state.pg_pool.acquire() as conn:
                await conn.execute(
                    "INSERT INTO command_history (user_id, instruction, status) VALUES ($1, $2, $3)",
                    payload.user_id,
                    payload.instruction,
                    "pending",
                )

        # 2. Initialize the Agent's state
        logger.info("🧠 Passing control to LangGraph Agent...")
        initial_state = {
            "instruction": payload.instruction,
            # Placeholder for future integration of real-time telemetry from Redis
            "current_telemetry": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
        }

        # 3. Execute the Graph (LangGraph handles Qdrant, LLM, and Routing)
        final_state = await agent_app.ainvoke(initial_state)

        # 4. Extract the final plan (guaranteed to be a list of PlanStep Pydantic models)
        plan_steps = final_state.get("final_plan", [])

        # Convert Pydantic models to standard dictionaries, omitting null fields (like explicit_goal)
        ros2_plan = [step.model_dump(exclude_none=True) for step in plan_steps]

        # 5. Queue task on Redis (Publisher) for the ROS 2 bridge
        if ros2_plan and state.redis_client:
            task_payload = {
                "task_id": point_id,
                "user_id": payload.user_id,
                "instruction": payload.instruction,
                "plan": ros2_plan,
            }

            queue_name = "robot_tasks_queue"
            await state.redis_client.rpush(queue_name, json.dumps(task_payload))
            logger.info(f"Task {point_id} successfully queued in Redis [{queue_name}].")

    except Exception as e:
        error_details = f"{type(e).__name__}: {str(e)}"
        logger.error(f"Error processing command from {payload.user_id}: {error_details}")
        raise HTTPException(status_code=500, detail=f"Internal processing error -> {error_details}")

    return CommandResponse(
        status="accepted",
        task_id=point_id,
        message="Command accepted, analyzed by Agent and sent to ROS 2 Fleet.",
        plan=ros2_plan,
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
                limit,
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
            "vector_preview": embedding_vector[:5],
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
            version = await conn.fetchval("SELECT version();")
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
            collection_name="semantic_memory", query=query_vector, limit=payload.limit
        ).points

        results = []
        for hit in search_result:
            if hit.payload is not None:
                results.append(
                    {
                        "score": round(hit.score, 4),
                        "instruction": hit.payload.get("instruction"),
                        "user_id": hit.payload.get("user_id"),
                        "pg_id": hit.payload.get("pg_id"),
                    }
                )

        return {"status": "success", "query": payload.query, "matches": results}
    except Exception as e:
        logger.error(f"Error during semantic search: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/llm-provider", tags=["System"])
async def set_llm_provider(payload: LLMProviderRequest):
    """Dynamically switch the LLM provider from the dashboard."""
    if payload.provider not in ["gemini", "ollama"]:
        raise HTTPException(
            status_code=400, detail="Invalid provider. Choose 'gemini' or 'ollama'."
        )

    state.llm_provider = payload.provider
    logger.info(f"🔄 LLM Provider switched to: {state.llm_provider}")

    return {"status": "success", "active_provider": state.llm_provider}


app.include_router(router)
