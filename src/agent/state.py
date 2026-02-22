###########################################################################
##                            IMPORTS
###########################################################################

from typing import Annotated, Literal

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


###########################################################################
##                         STATE SCHEMA
###########################################################################


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    depth: int
    max_iterations: int
    iteration: int
    collected_results: list[str]
    todo: list[str]
    reflection: str
    reflection_satisfied: bool
    rag_sources: list[dict]


###########################################################################
##                     STRUCTURED OUTPUT MODELS
###########################################################################


class RouteDecision(BaseModel):
    intent: Literal["direct_response", "needs_tools"] = Field(
        description="'direct_response' for simple chat, 'needs_tools' for data queries"
    )


class ReflectDecision(BaseModel):
    satisfied: bool = Field(description="True if collected data is sufficient to answer the question")
    feedback: str = Field(description="What additional data or queries would improve the answer")
    updated_todo: list[str] = Field(description="Updated remaining tasks, empty if satisfied")
