from langgraph.graph import END, StateGraph

from .nodes import node_format_dispatcher, node_planner, node_semantic_retriever
from .state import AgentState


def route_after_planning(state: AgentState):
    if not state.final_plan:
        return "dispatcher"

    # Ensure we iterate over a list, handling both Pydantic models and dictionaries
    plans = state.final_plan if isinstance(state.final_plan, list) else [state.final_plan]
    for step in plans:
        # Safely extract the action regardless of the internal object type
        action = step.get("action", "") if isinstance(step, dict) else getattr(step, "action", "")
        if "EXPLORE" in action:
            return "explore_node"

    return "dispatcher"


def build_agent_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("retriever", node_semantic_retriever)
    workflow.add_node("planner", node_planner)
    workflow.add_node("dispatcher", node_format_dispatcher)

    workflow.set_entry_point("retriever")
    workflow.add_edge("retriever", "planner")

    workflow.add_conditional_edges(
        "planner",
        route_after_planning,
        {
            "dispatcher": "dispatcher",
            # Temporary routing to dispatcher until explore_node is implemented
            # to prevent graph compilation errors.
            "explore_node": "dispatcher",
        },
    )

    workflow.add_edge("dispatcher", END)

    return workflow.compile()
