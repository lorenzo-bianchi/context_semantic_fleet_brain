import json

import httpx
from qdrant_client import models

from .state import AgentPlan, AgentState, PlanStep


async def node_semantic_retriever(state: AgentState) -> dict:
    from main import get_embedding
    from main import state as app_state

    if not app_state.qdrant_client:
        return {"semantic_context": []}

    try:
        query_vector = get_embedding(state.instruction)

        query_filter = models.Filter(
            must=[
                models.FieldCondition(key="type", match=models.MatchValue(value="visual_discovery"))
            ]
        )

        if state.inspected_point_ids:
            query_filter.must_not = [models.HasIdCondition(has_id=state.inspected_point_ids)]  # type: ignore[arg-type]

        search_result = app_state.qdrant_client.query_points(
            collection_name="semantic_memory",
            query=query_vector,
            limit=3,
            query_filter=query_filter,
        ).points

        context = []
        for hit in search_result:
            if hit.payload and hit.score > 0.23:
                context.append(
                    {
                        "id": hit.id,
                        "x": hit.payload.get("x"),
                        "y": hit.payload.get("y"),
                        "z": hit.payload.get("z"),
                        "yaw": hit.payload.get("yaw", 0.0),
                    }
                )

        return {"semantic_context": context}
    except Exception as e:
        return {"error_log": state.error_log + [f"Qdrant error: {str(e)}"]}


async def node_planner(state: AgentState) -> dict:
    from google.genai import types

    from main import state as app_state

    context_str = (
        json.dumps(state.semantic_context)
        if state.semantic_context
        else "No known objects in memory."
    )

    prompt = f"""You are the 'Fleet Brain', the AI of a ROS 2 robot.
    Analyze the command and extract the sequence of operations.
    Allowed actions: EXPLORE, NAVIGATE.

    Command: "{state.instruction}"
    Semantic Memory Context (Known objects): {context_str}

    CRITICAL RULES:
    1. If the user asks to "explore", "map", or "scan", output a single EXPLORE action.
    2. NAVIGATE ACTION: Use this ONLY if the requested object is explicitly listed in the Semantic Memory Context. You MUST include the "explicit_goal" array [x, y, z, yaw] AND the "point_id" string extracted from the context. Set "target" to the natural language name of the object.
    3. MISSING OBJECTS: If the user wants to go to an object but it is NOT in the Context (or the Context says "No known objects in memory."), you CANNOT use NAVIGATE. You MUST output an EXPLORE action to find it.

    EXAMPLES:
    Command: "Go to the red cube"
    Context: [{{"id": "...", "x": 2.0, "y": 1.0, "z": 0.5, "yaw": 0.0}}]
    {{
      "plan": [
        {{"action": "NAVIGATE", "target": "red cube", "explicit_goal": [2.0, 1.0, 0.5, 0.0]}}
      ]
    }}

    Command: "Reach the blue sphere"
    Context: No known objects in memory.
    {{
      "plan": [
        {{"action": "EXPLORE", "target": "environment"}}
      ]
    }}
    """

    try:
        if app_state.llm_provider == "ollama":
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    app_state.ollama_url,
                    json={
                        "model": "gemma2:9b",
                        "prompt": prompt,
                        "stream": False,
                        "format": AgentPlan.model_json_schema(),
                    },
                    timeout=60.0,
                )
                response.raise_for_status()
                raw_json = response.json().get("response", "{}")
                parsed_data = AgentPlan.model_validate_json(raw_json)
                plan = parsed_data.plan
        else:
            response = app_state.gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=AgentPlan,
                ),
            )
            parsed_data = AgentPlan.model_validate_json(response.text)
            plan = parsed_data.plan

        return {"final_plan": plan}

    except Exception as e:
        fallback_step = PlanStep(action="EXPLORE", target="environment")
        return {
            "error_log": state.error_log + [f"Planner error: {str(e)}"],
            "final_plan": [fallback_step],
        }


async def node_format_dispatcher(state: AgentState) -> dict:
    if not state.final_plan:
        fallback = PlanStep(action="EXPLORE", target="environment")
        return {"final_plan": [fallback]}

    return {"final_plan": state.final_plan}
