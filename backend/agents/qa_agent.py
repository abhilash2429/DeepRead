from __future__ import annotations

import os
from typing import TypedDict

from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

from backend.agents.code_agent import generate_component_code
from backend.models.briefing import InternalRepresentation
from backend.models.paper import ParsedPaper
from backend.prompts.qa import QA_PROMPT
from backend.tools import build_analysis_tools, build_code_tools, build_knowledge_tools, build_paper_tools


QA_MODEL = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.2,
)


class QAState(TypedDict, total=False):
    user_message: str
    response: str
    messages: list[BaseMessage]


def _build_executor(
    parsed_paper: ParsedPaper,
    internal_rep: InternalRepresentation,
    briefing_markdown: str,
    resolved_ambiguities: dict[str, str] | None = None,
) -> AgentExecutor:
    resolved_ambiguities = resolved_ambiguities or {}

    async def _code_callback(component_name: str) -> str:
        snippet = await generate_component_code(
            component_name=component_name,
            component_description=f"Generated at user request for {component_name}",
            relevant_sections=[component_name],
            relevant_equations=[],
            resolved_ambiguities=resolved_ambiguities,
        )
        return snippet.code

    tools = [
        *build_paper_tools(parsed_paper),
        *build_knowledge_tools(internal_rep),
        *build_analysis_tools(internal_rep),
        *build_code_tools(_code_callback),
    ]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"{QA_PROMPT}\n\n"
                f"Briefing content:\n{briefing_markdown}\n\n"
                f"InternalRepresentation JSON:\n{internal_rep.model_dump_json(indent=2)}",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )
    agent = create_tool_calling_agent(QA_MODEL, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True, verbose=False)


def _build_qa_graph(executor: AgentExecutor):
    async def qa_node(state: QAState) -> QAState:
        chat_history = list(state.get("messages", []))
        result = await executor.ainvoke({"input": state["user_message"], "chat_history": chat_history[-12:]})
        output = str(result.get("output", "")).strip()
        state["response"] = output
        state["messages"] = [*chat_history, HumanMessage(content=state["user_message"]), AIMessage(content=output)]
        return state

    workflow = StateGraph(QAState)
    workflow.add_node("qa", qa_node)
    workflow.set_entry_point("qa")
    workflow.add_edge("qa", END)
    return workflow.compile()


async def run_qa_turn(
    user_message: str,
    parsed_paper: ParsedPaper,
    internal_rep: InternalRepresentation,
    briefing_markdown: str,
    chat_history: list[BaseMessage],
    resolved_ambiguities: dict[str, str] | None = None,
) -> tuple[str, list[BaseMessage]]:
    executor = _build_executor(
        parsed_paper=parsed_paper,
        internal_rep=internal_rep,
        briefing_markdown=briefing_markdown,
        resolved_ambiguities=resolved_ambiguities,
    )
    graph = _build_qa_graph(executor)
    result = await graph.ainvoke({"user_message": user_message, "messages": list(chat_history)})
    return result.get("response", ""), result.get("messages", list(chat_history))
