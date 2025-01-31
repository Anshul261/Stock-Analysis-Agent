from tools.yahoo_finance import yahoo_finance_tool
import os
from pydantic import BaseModel
from typing import TypedDict, Annotated, Union, Literal, Any
from dotenv import load_dotenv
import mlflow
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages, AnyMessage
from langchain_core.tools import tool
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_openai import AzureChatOpenAI

load_dotenv()
mlflow.langchain.autolog()

# Define tools
file_management_tools = FileManagementToolkit(
    root_dir=str("./"),
    selected_tools=["read_file", "write_file", "list_directory"],
).get_tools()
read_tool, write_tool, list_tool = file_management_tools

def tools_condition(
    state: Union[list[AnyMessage], dict[str, Any], BaseModel],
    messages_key: str = "messages",
) -> Literal["tools", END]:
    """Use in the conditional_edge to route to the ToolNode if the last message

    has tool calls. Otherwise, route to the end.

    Args:
        state (Union[list[AnyMessage], dict[str, Any], BaseModel]): The state to check for
            tool calls. Must have a list of messages (MessageGraph) or have the
            "messages" key (StateGraph).

    Returns:
        The next node to route to.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif isinstance(state, dict) and (messages := state.get(messages_key, [])):
        ai_message = messages[-1]
    elif messages := getattr(state, messages_key, []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END

llm = AzureChatOpenAI(
deployment_name="gpt-4",
model_name="gpt-4",
azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
api_version=os.getenv('OPENAI_API_VERSION'),
openai_api_key=os.getenv('AZURE_OPENAI_API_KEY'),
)
tools = [yahoo_finance_tool, read_tool, write_tool]

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)
graph_builder.set_entry_point("chatbot")
graph_builder.add_node("tools", ToolNode(tools))
graph_builder.add_node("chatbot", lambda state: {"messages":llm.bind_tools(tools).invoke(state['messages'])})
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_conditional_edges(
    "chatbot", tools_condition
)
graph = graph_builder.compile()
# from generate_graph import generate
# generate(graph)

# PKTM.SW
stock = "NVDA"

prompt = f"""You are a financial analyst:
 - use yfinance tool to get financial data from stock {stock} 
 - use read_tool to read Warrent Buffet principles from file warren_buffet_investement_principles.txt
 - using those principles, do a stock analysis recommandation, rate from 1 to 10 the stock
 - using the write_tool, write into a markdown file analysis.md the analysis"""

print(graph.invoke({"messages": {"role": "user", "content": prompt}}))