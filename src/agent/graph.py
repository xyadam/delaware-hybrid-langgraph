###########################################################################
##                          PATH SETUP
###########################################################################

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

###########################################################################
##                            IMPORTS
###########################################################################

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

###########################################################################
##                        CUSTOM IMPORTS
###########################################################################

from agent.state import AgentState
from agent.shared import TOOLS
from agent.nodes import (
    router, plan, collect_results, reflect, synthesize, respond,
    route_after_router, route_after_plan, route_after_reflect,
)


###########################################################################
##                        GRAPH DEFINITION
###########################################################################

workflow = StateGraph(AgentState)

# Nodes
workflow.add_node("router", router)
workflow.add_node("respond", respond)
workflow.add_node("plan", plan)
workflow.add_node("execute", ToolNode(TOOLS))
workflow.add_node("collect_results", collect_results)
workflow.add_node("reflect", reflect)
workflow.add_node("synthesize", synthesize)

# Edges
workflow.add_edge(START, "router")
workflow.add_conditional_edges("router", route_after_router, {"respond": "respond", "plan": "plan"})
workflow.add_conditional_edges("plan", route_after_plan, {"execute": "execute", "synthesize": "synthesize"})
workflow.add_edge("execute", "collect_results")
workflow.add_edge("collect_results", "reflect")
workflow.add_conditional_edges("reflect", route_after_reflect, {"synthesize": "synthesize", "plan": "plan"})
workflow.add_edge("respond", END)
workflow.add_edge("synthesize", END)

# Module-level compiled graph (no checkpointer — langgraph dev/Studio provides its own)
agent = workflow.compile()
