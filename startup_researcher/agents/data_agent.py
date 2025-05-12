# startup_researcher/agents/data_agent.py

import asyncio
import os
import pathlib
from typing import Annotated, cast

import chainlit as cl
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain_core.runnables.config import RunnableConfig
from langchain_mcp_adapters.client import MultiServerMCPClient, StdioServerParameters
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode, create_react_agent

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---

# Ensure necessary environment variables are set
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH")

if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY environment variable not set.")
# We will check SQLITE_DB_PATH in on_chat_start to provide a better error message in the UI

# Define the directory where the agent script is located to resolve relative paths
SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()

# --- MCP Server Definitions ---


def get_mcp_server_configs(db_path: str) -> dict[str, StdioServerParameters]:
    """Defines the configurations for the MCP servers."""
    servers = {}

    # 1. SQLite Server Configuration
    # Requires: npx installed, internet connection (for initial npx download)
    # The database path needs to be absolute for the server process.
    absolute_db_path = str(pathlib.Path(db_path).resolve())
    servers["sqlite"] = StdioServerParameters(
        # Use npx to run the package directly. Ensure Node.js/npx is installed.
        command="npx",
        args=[
            "-y",  # Auto-confirm installation
            "@modelcontextprotocol/sqlite-mcp-server",  # The official SQLite MCP server package
            "stdio",  # Use stdio transport
            "--db",
            absolute_db_path,  # Pass the absolute database path
        ],
        transport="stdio",
    )

    # 2. Python Execution Server Configuration (Optional but useful for analysis/viz)
    # Requires: deno installed, internet connection (for initial deno download)
    servers["python_runner"] = StdioServerParameters(
        command="deno",
        args=[
            "run",
            "-A",  # Allow all permissions (adjust if needed for security)
            "--node-modules-dir=auto",
            "jsr:@pydantic/mcp-run-python",  # Pydantic's Python runner
            "stdio",  # Use stdio transport
        ],
        transport="stdio",
    )
    return servers


# --- LangGraph Agent Setup ---


# Using MessagesState for conversation history
class AgentState(MessagesState):
    pass


# --- Chainlit Integration ---


@cl.on_chat_start
async def on_chat_start():
    """Initializes the agent and MCP client when a new chat session starts."""
    if not SQLITE_DB_PATH:
        await cl.Message(
            content=
            "Error: SQLITE_DB_PATH environment variable is not set. Please configure the path to your SQLite database in your .env file."
        ).send()
        return
    if not pathlib.Path(SQLITE_DB_PATH).exists():
        await cl.Message(
            content=
            f"Error: SQLite database not found at path: {SQLITE_DB_PATH}. Please ensure the path is correct and the file exists."
        ).send()
        return

    try:
        # Configure MCP servers
        server_configs = get_mcp_server_configs(SQLITE_DB_PATH)
        mcp_client = MultiServerMCPClient(server_configs)

        # Start the MCP client and servers
        # The client needs to be active for the duration of the session
        # We manage its lifecycle using __aenter__ and __aexit__
        await mcp_client.__aenter__()
        cl.user_session.set("mcp_client", mcp_client)

        # Get tools from the active MCP client
        tools = mcp_client.get_tools()
        if not tools:
            await cl.Message(
                content=
                "Error: Failed to retrieve tools from MCP servers. Check server logs or configurations."
            ).send()
            await mcp_client.__aexit__(None, None, None)  # Clean up
            return

        # Initialize the LLM (Claude 3.5 Sonnet recommended for tool use)
        model = ChatAnthropic(model="claude-3.5-sonnet-20240620",
                              temperature=0)
        model_with_tools = model.bind_tools(tools)

        # Create the ReAct agent using LangGraph prebuilt function
        # Pass the model with tools bound and the retrieved MCP tools
        agent_runnable = create_react_agent(model_with_tools, tools)

        cl.user_session.set("agent_runnable", agent_runnable)
        await cl.Message(
            content=
            f"Data Agent connected to database: `{SQLITE_DB_PATH}`. How can I help you analyze the data?"
        ).send()

    except Exception as e:
        await cl.Message(content=f"Error during agent initialization: {str(e)}"
                         ).send()
        # Attempt cleanup if client was partially started
        client = cl.user_session.get("mcp_client")
        if client:
            try:
                await client.__aexit__(None, None, None)
            except Exception as cleanup_e:
                print(f"Error during MCP client cleanup: {cleanup_e}"
                      )  # Log cleanup error


@cl.on_message
async def on_message(message: cl.Message):
    """Handles incoming user messages and runs the agent."""
    agent_runnable = cl.user_session.get("agent_runnable")
    mcp_client = cl.user_session.get(
        "mcp_client")  # Retrieve the active client

    if not agent_runnable or not mcp_client:
        await cl.Message(
            content="Agent not initialized. Please check the logs.").send()
        return

    msg = cl.Message(content="")
    await msg.send()  # Send an empty message to indicate processing

    # The agent needs the client context to be active for tool calls.
    # Since we started the client in on_chat_start and keep it active,
    # the tools obtained via get_tools() should function correctly.
    try:
        # Use the LangchainCallbackHandler to stream intermediate steps
        config = RunnableConfig(
            callbacks=[cl.LangchainCallbackHandler(stream_final_answer=True)],
            # Define a unique thread_id for memory (if using persistence)
            configurable={"thread_id": cl.context.session.thread_id},
        )

        # Stream the agent's response
        async for chunk in agent_runnable.astream(
            {"messages": [HumanMessage(content=message.content)]},
            config=config,
        ):
            # Agent state chunks - we only care about the final message stream
            if "messages" in chunk:
                # The last message is the response
                await msg.stream_token(chunk["messages"][-1].content)

        await msg.update()

    except Exception as e:
        await cl.Message(content=f"An error occurred: {str(e)}").send()


@cl.on_chat_end
async def on_chat_end():
    """Cleans up resources when the chat session ends."""
    mcp_client = cl.user_session.get("mcp_client")
    if mcp_client:
        try:
            print("Shutting down MCP client...")
            await mcp_client.__aexit__(None, None, None)
            print("MCP client shut down.")
        except Exception as e:
            print(f"Error shutting down MCP client: {e}")  # Log error
