from __future__ import annotations

import json
from typing import Any, Literal, TypedDict

from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph
from pydantic import BaseModel

from backend.agents.code_agent import generate_component_code
from backend.models.artifacts import CodeSnippet
from backend.models.conversation import ConversationState, InternalRepresentation, Stage
from backend.models.paper import ParsedPaper
from backend.prompts.stages import STAGE_PROMPTS
from backend.services.gemini_client import GeminiClient
from backend.tools import build_analysis_tools, build_code_tools, build_knowledge_tools, build_paper_tools


class RouteDecision(BaseModel):
    intent: Literal[
        "continue_current_stage",
        "orientation",
        "architecture",
        "implementation",
        "ambiguity",
        "training",
        "freeqa",
    ]


class PaperLensState(TypedDict, total=False):
    session_id: str
    user_message: str
    stage_override: str | None
    current_stage: str
    route_key: str
    internal_rep: InternalRepresentation
    parsed_paper: ParsedPaper
    resolved_ambiguities: dict[str, str]
    messages: list[BaseMessage]
    response_text: str
    generated_snippets: list[CodeSnippet]
    pending_question: str | None


def _guess_component(internal_rep: InternalRepresentation, user_message: str) -> str | None:
    needle = user_message.lower()
    components = {e.parent for e in internal_rep.component_graph} | {e.child for e in internal_rep.component_graph}
    for component in sorted(components, key=len, reverse=True):
        if component.lower() in needle:
            return component
    if "attention" in needle:
        for component in components:
            if "attention" in component.lower():
                return component
    return None


def _unresolved_ambiguity_question(internal_rep: InternalRepresentation) -> str | None:
    for amb in internal_rep.ambiguity_log:
        if not amb.resolved:
            return (
                f"Ambiguity {amb.ambiguity_id}: {amb.ambiguous_point}\n"
                f"Impact: {amb.implementation_consequence}\n"
                f"Best guess: {amb.best_guess_resolution}\n"
                "What decision do you want to lock in?"
            )
    return None


async def _classify_route(gemini: GeminiClient, user_message: str, current_stage: str, stage_override: str | None) -> str:
    if stage_override and stage_override in STAGE_PROMPTS:
        return stage_override

    parser = PydanticOutputParser(pydantic_object=RouteDecision)
    prompt = ChatPromptTemplate.from_template(
        "Classify user intent for an ML paper tutoring assistant.\n"
        "Current stage: {current_stage}\n"
        "User message: {user_message}\n\n"
        "Return intent JSON.\n{format_instructions}"
    )
    chain = prompt | gemini.llm | parser
    try:
        decision = await chain.ainvoke(
            {
                "current_stage": current_stage,
                "user_message": user_message,
                "format_instructions": parser.get_format_instructions(),
            }
        )
        if decision.intent == "continue_current_stage":
            return current_stage
        if decision.intent == "freeqa":
            return "freeqa"
        return decision.intent
    except Exception:
        msg = user_message.lower()
        if any(k in msg for k in ["code", "implement", "pytorch", "snippet"]):
            return Stage.IMPLEMENTATION.value
        if any(k in msg for k in ["ambiguity", "assumption", "underspecified"]):
            return Stage.AMBIGUITY.value
        if any(k in msg for k in ["train", "optimizer", "hyperparameter", "batch size"]):
            return Stage.TRAINING.value
        if any(k in msg for k in ["architecture", "equation", "component", "attention"]):
            return Stage.ARCHITECTURE.value
        if any(k in msg for k in ["overview", "summary", "orientation", "novelty"]):
            return Stage.ORIENTATION.value
        return current_stage


def _stage_agent_executor(
    gemini: GeminiClient,
    stage: str,
    tools: list[Any],
    internal_rep: InternalRepresentation,
) -> AgentExecutor:
    system = (
        f"{STAGE_PROMPTS.get(stage, STAGE_PROMPTS[Stage.ORIENTATION.value])}\n\n"
        "You must provide implementation-usable detail and cite section/equation references.\n"
        "If something is unstated, label it as inferred or assumed explicitly.\n"
        f"InternalRepresentation JSON:\n{internal_rep.model_dump_json()}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_tool_calling_agent(gemini.llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=False, handle_parsing_errors=True)


def build_conversation_graph(gemini: GeminiClient):
    async def router_node(state: PaperLensState) -> PaperLensState:
        route_key = await _classify_route(
            gemini=gemini,
            user_message=state["user_message"],
            current_stage=state["current_stage"],
            stage_override=state.get("stage_override"),
        )
        if route_key in STAGE_PROMPTS:
            state["current_stage"] = route_key
        state["route_key"] = route_key
        return state

    async def run_stage_node(state: PaperLensState) -> PaperLensState:
        internal_rep = state["internal_rep"]
        parsed_paper = state["parsed_paper"]
        chat_history = state.get("messages", [])
        stage = state["current_stage"]

        async def _code_for_component(component_name: str) -> str:
            snippet = await generate_component_code(
                gemini=gemini,
                component_name=component_name,
                component_description=f"Implementation details for {component_name}",
                relevant_sections=[component_name],
                relevant_equations=[],
                resolved_ambiguities=state.get("resolved_ambiguities", {}),
            )
            return snippet.code

        tools: list[Any] = []
        tools.extend(build_paper_tools(parsed_paper))
        tools.extend(build_knowledge_tools(internal_rep))
        tools.extend(build_analysis_tools(internal_rep))
        tools.extend(build_code_tools(_code_for_component))

        executor = _stage_agent_executor(gemini=gemini, stage=stage, tools=tools, internal_rep=internal_rep)
        try:
            result = await executor.ainvoke({"input": state["user_message"], "chat_history": chat_history[-10:]})
            response = str(result.get("output", "")).strip()
        except Exception:
            fallback_prompt = (
                f"{STAGE_PROMPTS.get(stage, STAGE_PROMPTS[Stage.ORIENTATION.value])}\n\n"
                "Provide a clear, implementation-focused answer.\n"
                f"User message: {state['user_message']}\n"
            )
            response = (await gemini.generate_text(fallback_prompt)).strip()
        snippets: list[CodeSnippet] = []
        pending_question: str | None = None

        if stage == Stage.IMPLEMENTATION.value:
            component = _guess_component(internal_rep, state["user_message"]) or "MainModel"
            snippet = await generate_component_code(
                gemini=gemini,
                component_name=component,
                component_description=f"Implementation-focused explanation for {component}",
                relevant_sections=[component],
                relevant_equations=[],
                resolved_ambiguities=state.get("resolved_ambiguities", {}),
            )
            snippets.append(snippet)
            response = f"{response}\n\n```python\n{snippet.code}\n```"

        if stage == Stage.AMBIGUITY.value:
            pending_question = _unresolved_ambiguity_question(internal_rep)
            if pending_question:
                response = f"{response}\n\n{pending_question}"

        state["response_text"] = response
        state["generated_snippets"] = snippets
        state["pending_question"] = pending_question
        state["messages"] = [*chat_history, HumanMessage(content=state["user_message"]), AIMessage(content=response)]
        return state

    workflow = StateGraph(PaperLensState)
    workflow.add_node("router", router_node)
    workflow.add_node("orientation", run_stage_node)
    workflow.add_node("architecture", run_stage_node)
    workflow.add_node("implementation", run_stage_node)
    workflow.add_node("ambiguity", run_stage_node)
    workflow.add_node("training", run_stage_node)
    workflow.add_node("freeqa", run_stage_node)

    workflow.set_entry_point("router")
    workflow.add_conditional_edges(
        "router",
        lambda s: s.get("route_key", Stage.ORIENTATION.value),
        {
            "orientation": "orientation",
            "architecture": "architecture",
            "implementation": "implementation",
            "ambiguity": "ambiguity",
            "training": "training",
            "freeqa": "freeqa",
        },
    )
    for node in ["orientation", "architecture", "implementation", "ambiguity", "training", "freeqa"]:
        workflow.add_edge(node, END)
    return workflow.compile()


async def run_graph_turn(
    gemini: GeminiClient,
    conversation_state: ConversationState,
    parsed_paper: ParsedPaper,
    user_message: str,
    stage_override: str | None = None,
    chat_history: list[BaseMessage] | None = None,
) -> tuple[str, list[CodeSnippet], str | None, str, list[BaseMessage]]:
    graph = build_conversation_graph(gemini)
    initial_state: PaperLensState = {
        "session_id": conversation_state.session_id,
        "user_message": user_message,
        "stage_override": stage_override,
        "current_stage": conversation_state.current_stage.value,
        "internal_rep": conversation_state.internal_representation,
        "parsed_paper": parsed_paper,
        "resolved_ambiguities": dict(conversation_state.resolved_ambiguities),
        "messages": list(chat_history or []),
    }
    result: PaperLensState = await graph.ainvoke(initial_state)
    return (
        result.get("response_text", ""),
        result.get("generated_snippets", []),
        result.get("pending_question"),
        result.get("current_stage", conversation_state.current_stage.value),
        result.get("messages", []),
    )


async def stream_graph_events(gemini: GeminiClient, state: PaperLensState):
    graph = build_conversation_graph(gemini)
    async for event in graph.astream_events(state, version="v2"):
        yield event
