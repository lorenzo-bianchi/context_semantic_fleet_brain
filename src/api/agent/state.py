from typing import Any

from pydantic import BaseModel, Field


class PlanStep(BaseModel):
    action: str = Field(description="The action to perform: EXPLORE or NAVIGATE")
    target: str = Field(description="The semantic target or 'coordinates'")
    explicit_goal: list[float] | None = Field(default=None, description="Optional [x, y, z, yaw]")
    point_id: str | None = Field(
        default=None, description="The Qdrant ID if the object is from memory"
    )


class AgentPlan(BaseModel):
    plan: list[PlanStep]


class AgentState(BaseModel):
    instruction: str = Field(..., description="The original user command")
    current_telemetry: dict[str, float] = Field(
        default_factory=dict, description="Current position of the drone (x, y, z, yaw)"
    )
    semantic_context: list[Any] = Field(
        default_factory=list, description="Results extracted from the Qdrant vector memory"
    )
    vision_context: str | None = Field(
        default=None, description="Descriptions processed by the CLIP model"
    )
    inspected_point_ids: list[str] = Field(
        default_factory=list, description="List of Qdrant IDs already inspected and rejected"
    )
    error_log: list[str] = Field(
        default_factory=list, description="Internal log to handle fallbacks and failures"
    )
    final_plan: list[PlanStep] = Field(
        default_factory=list, description="Final list of actions for the ROS 2 bridge"
    )
