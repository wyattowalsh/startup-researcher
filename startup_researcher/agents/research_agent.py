import asyncio
import os
from typing import Annotated, Any, Dict, List, Optional, TypedDict

import yaml
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI  # Or ChatAnthropic, etc.
from langgraph.checkpoint.memory import (
    MemorySaver,  # For now, can be swapped for persistent
)
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from startup_researcher.config import settings
from startup_researcher.logging import logger
from startup_researcher.models import Company as CompanyPydantic
from startup_researcher.utils import init_db, upsert_company_data


# Define the state for the research agent graph
class ResearchState(TypedDict):
    startup_query: str  # Initial query/startup description
    company_name_to_research: Optional[
        str]  # Extracted or refined company name
    search_queries: List[str]  # List of search queries to execute
    search_results: Dict[str, List[Dict[
        str, Any]]]  # Results from search tools, keyed by query
    urls_to_fetch: List[str]  # URLs identified for fetching content
    fetched_content: Dict[str, str]  # Content fetched from URLs, keyed by URL
    # Raw extracted data from LLM, can be a dict before Pydantic validation
    extracted_raw_data: Optional[Dict[str, Any]]
    # Validated Pydantic model of the company data
    extracted_company_data: Optional[CompanyPydantic]
    # Log of operations and decisions
    research_log: List[str]
    # All messages in the conversation with tools and agent
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]
    # Store the final CompanyOrm object if successfully saved
    saved_company_id: Optional[int]
    # Next step decision
    _next_step_decision: Optional[str]


class ResearchAgent:

    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.OPENAI_MODEL_NAME,
            temperature=0.1,
            # api_key=settings.OPENAI_API_KEY # Handled by env var usually
        )
        # MCP Client setup will be done asynchronously in the `run` method
        self.mcp_client: Optional[MultiServerMCPClient] = None
        self.tools: Optional[List[BaseTool]] = None

        # Initialize graph
        workflow = StateGraph(ResearchState)

        # Define nodes
        workflow.add_node("plan_research", self._plan_research)
        workflow.add_node("execute_searches",
                          self._execute_searches)  # This will be a ToolNode
        workflow.add_node("analyze_search_results",
                          self._analyze_search_results)
        workflow.add_node("fetch_content",
                          self._fetch_content)  # This will be a ToolNode
        workflow.add_node("extract_information", self._extract_information)
        workflow.add_node("validate_and_save_data",
                          self._validate_and_save_data)
        workflow.add_node("handle_error", self._handle_error)

        # Define edges
        workflow.add_edge(START, "plan_research")
        workflow.add_conditional_edges(
            "plan_research", self._decide_after_planning, {
                "execute_searches": "execute_searches",
                "fetch_content": "fetch_content",
                "extract_information": "extract_information",
                "end_research": END
            })
        workflow.add_edge("execute_searches", "analyze_search_results")
        workflow.add_conditional_edges(
            "analyze_search_results", self._decide_after_search_analysis, {
                "fetch_content": "fetch_content",
                "extract_information": "extract_information",
                "plan_research": "plan_research",
                "end_research": END
            })
        workflow.add_edge("fetch_content", "extract_information"
                          )  # Fetched content directly leads to extraction
        workflow.add_conditional_edges(
            "extract_information", self._decide_after_extraction, {
                "validate_and_save_data": "validate_and_save_data",
                "plan_research": "plan_research",
                "handle_error": "handle_error"
            })
        workflow.add_edge("validate_and_save_data", END)
        workflow.add_edge("handle_error",
                          END)  # Or loop back to plan_research for retry

        self.graph = workflow.compile(checkpointer=MemorySaver())
        logger.info("Research agent graph compiled.")

    async def _setup_mcp_client_and_tools(self):
        if self.mcp_client is None:
            # Ensure API keys are set in environment or via settings
            brave_api_key = getattr(settings, 'BRAVE_API_KEY',
                                    'YOUR_API_KEY_HERE')
            tavily_api_key = getattr(settings, 'TAVILY_API_KEY', None)
            fetch_script_path = getattr(
                settings, 'FETCH_MCP_SCRIPT_PATH',
                './fetch-mcp/dist/index.js')  # Example default

            if not tavily_api_key:
                logger.error(
                    "TAVILY_API_KEY is not set in environment/settings.")
                # Handle error appropriately - maybe raise exception or disable tool
                # For now, we'll proceed but Tavily tool might fail

            if brave_api_key == 'YOUR_API_KEY_HERE':
                logger.warning(
                    "BRAVE_API_KEY is not set or using placeholder.")

            if not os.path.exists(fetch_script_path):
                logger.error(
                    f"Fetch MCP script not found at: {fetch_script_path}. "
                    f"Please build fetch-mcp and set FETCH_MCP_SCRIPT_PATH.")
                # Handle error - potentially disable fetch tool

            self.mcp_client = MultiServerMCPClient({
                # Tool name used in code: tavily_search
                "tavily_search": {
                    "command": "npx",
                    "args": ["-y", "tavily-mcp@0.1.4"],  # Use specific version
                    "transport": "stdio",
                    "env": {
                        "TAVILY_API_KEY": tavily_api_key or ""
                    }
                },
                # Tool name used in code: brave_search
                "brave_search": {
                    "command": "npx",
                    "args":
                    ["-y", "@modelcontextprotocol/server-brave-search"],
                    "transport": "stdio",
                    "env": {
                        "BRAVE_API_KEY": brave_api_key
                    }
                },
                # Tool name used in code: fetch_web_content
                "fetch_web_content": {
                    "command": "node",
                    "args": [
                        fetch_script_path  # Use configured path
                    ],
                    "transport": "stdio",
                    # Fetch might not need env vars here if not configured in script
                }
            })
            # Changed from start_servers to start()
            await self.mcp_client.start()
            self.tools = self.mcp_client.get_tools()
            logger.info(
                f"MCP Client started and tools loaded: {[tool.name for tool in self.tools]}"
            )
            # Bind tools to LLM for function calling
            self.llm = self.llm.bind_tools(self.tools)

    # --- Node Implementations ---
    async def _plan_research(self, state: ResearchState) -> ResearchState:
        logger.info(f"Planning research for: {state['startup_query']}")
        log_entry = (f"Starting research plan for query: "
                     f"'{state['startup_query']}'")
        current_log = state.get("research_log", [])
        messages = state.get("messages", [])

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             """You are a highly intelligent research planning assistant.
            Given a startup query, your goal is to devise a research plan. This involves:
            1. Identifying the primary company name to research from the query.
            2. Formulating a list of targeted search queries for Tavily and Brave Search to gather comprehensive information about the company (e.g., founding, funding, products, team, market, news).
            3. If initial information is very sparse, suggest broader search terms.
            4. Based on prior research steps (if any, from messages), refine the plan.

            Consider what information is needed to fill a comprehensive company profile including:
            - Basic Info (website, founding, HQ)
            - Founders, Funding Rounds, Products/Technologies, Patents, Market Segments
            - Revenue Models, Pricing, GTM, Competitors, SWOT
            - Press Releases, Growth Metrics, Compliance News, Executives, Culture, Risks, Opportunities.

            Output a JSON object with keys: 'company_name_to_research', 'search_queries' (list of strings).
            If previous searches yielded specific URLs of high interest that haven't been fetched, you can also include 'urls_to_fetch' (list of strings).
            If you believe enough information might have been gathered for a first pass extraction, you can suggest proceeding to extraction by setting 'proceed_to_extraction': true.
            If you believe the research is complete, set 'research_complete': true.
            """),
            ("human",
             "Startup Query: {startup_query}\n\nPrior Research Log:\n{research_log_str}\n\nConversation History:\n{history_str}"
             )
        ])
        chain = prompt | self.llm
        response = await chain.ainvoke({
            "startup_query":
            state['startup_query'],
            "research_log_str":
            "\n".join(current_log[-5:]),  # last 5 log entries for context
            "history_str":
            "\n".join(
                [f"{m.type}: {m.content[:200]}..." for m in messages[-5:]])
        })

        messages.append(response)  # Add LLM's response to history
        plan_data = {}  # Initialize with empty dict
        next_step = "handle_error"  # Default to error if parsing fails
        try:
            # Ensure response.content is a string before trying to parse if it's not already a dict
            content_to_parse = response.content
            if isinstance(response.content, str):
                # A simple heuristic to find a JSON block if the LLM wraps it.
                # More robust parsing might be needed for complex LLM outputs.
                json_start = content_to_parse.find('{')
                json_end = content_to_parse.rfind('}') + 1
                if json_start != -1 and json_end != -1:
                    content_to_parse = content_to_parse[json_start:json_end]
                plan_data = json.loads(content_to_parse)
            elif isinstance(
                    response.content, dict
            ):  # If LLM directly outputs a dict (e.g. with OpenAI function calling for plan)
                plan_data = response.content
            else:
                logger.warning(
                    f"LLM plan response was not a string or dict: {type(response.content)}"
                )

            log_entry += f"\nLLM Plan: {plan_data}"
            logger.info(f"LLM Plan for {state['startup_query']}: {plan_data}")

            updated_state = state.copy()
            updated_state["company_name_to_research"] = plan_data.get(
                "company_name_to_research",
                state.get("company_name_to_research"))
            updated_state["search_queries"] = plan_data.get(
                "search_queries", [])
            updated_state["urls_to_fetch"] = plan_data.get("urls_to_fetch", [])
            updated_state["research_log"] = current_log + [log_entry]
            updated_state["messages"] = messages

            # Decision logic moved here
            if plan_data.get("research_complete"):
                next_step = "end_research"
            elif plan_data.get("proceed_to_extraction") and state.get(
                    "fetched_content"):
                next_step = "extract_information"
            elif updated_state["search_queries"]:
                next_step = "execute_searches"
            elif updated_state["urls_to_fetch"]:
                next_step = "fetch_content"
            else:
                next_step = "extract_information"  # Default if no other actions

            updated_state[
                "_next_step_decision"] = next_step  # Store decision for conditional edge

            return updated_state
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse plan JSON from LLM: {e}. "
                         f"Response: {response.content}")
            log_entry += f"\nError parsing LLM plan: {e}"
            messages.append(
                AIMessage(
                    content=(f"Error: Could not parse planning step output. "
                             f"Will try to extract based on current info. "
                             f"Raw plan: {response.content}")))
            return {
                **state, "research_log": current_log + [log_entry],
                "messages": messages,
                "_next_step_decision": "handle_error"
            }
        except Exception as e:
            logger.error(f"Unexpected error in _plan_research: {e}")
            log_entry += f"\nUnexpected error in planning: {e}"
            messages.append(
                AIMessage(content=(f"Error: Unexpected error during planning. "
                                   f"Raw plan: {response.content}")))
            return {
                **state, "research_log": current_log + [log_entry],
                "messages": messages,
                "_next_step_decision": "handle_error"
            }

    async def _execute_searches(self, state: ResearchState) -> ResearchState:
        logger.info(
            f"Executing searches for: {state.get('company_name_to_research')}")
        log_entry = "Executing searches..."
        current_log = state.get("research_log", [])
        messages = state.get("messages", [])
        tool_invocations = []

        # Prefer Tavily first, then Brave if specific tools are named or chosen by LLM later
        # For now, let's assume the LLM plan node could specify which tool to use per query or we use a default
        # This node is simplified: it takes search_queries and executes them using available search tools.
        # A more complex version would involve the LLM choosing *which* search tool to use from self.tools.

        # This node is now a ToolNode. We need to format the message for it if the LLM didn't directly output tool_calls.
        # For simplicity, if state['search_queries'] is populated by the planner, we manually create tool calls.

        company_name = state.get("company_name_to_research",
                                 state['startup_query'])
        search_results_agg = state.get("search_results", {})

        if not self.tools:
            log_entry += " Error: MCP tools not initialized."
            logger.error("MCP tools not initialized in _execute_searches")
            messages.append(
                AIMessage(content="Error: Search tools are not available."))
            return {
                **state, "research_log": current_log + [log_entry],
                "messages": messages,
                "_next_step_decision": "handle_error"
            }

        # Find Tavily and Brave tools
        tavily_tool = next(
            (t for t in self.tools if "tavily" in t.name.lower()), None)
        brave_tool = next((t for t in self.tools if "brave" in t.name.lower()),
                          None)

        if not tavily_tool and not brave_tool:
            log_entry += " Error: Neither Tavily nor Brave search tools are available."
            logger.error("Neither Tavily nor Brave search tools found.")
            messages.append(
                AIMessage(
                    content="Error: No suitable search tools available."))
            return {
                **state, "research_log": current_log + [log_entry],
                "messages": messages,
                "_next_step_decision": "handle_error"
            }

        for query_idx, query_text in enumerate(state.get("search_queries",
                                                         [])):
            # Alternate between Tavily and Brave, or use a preferred one if available
            tool_to_use = None
            if query_idx % 2 == 0 and tavily_tool:  # Prioritize Tavily for even queries
                tool_to_use = tavily_tool
            elif brave_tool:
                tool_to_use = brave_tool
            elif tavily_tool:  # Fallback to Tavily if Brave wasn't chosen/available
                tool_to_use = tavily_tool
            else:
                # Should not happen due to check above, but as a safeguard
                logger.warning(
                    f"No search tool could be selected for query: {query_text}"
                )
                continue

            logger.info(f"Using {tool_to_use.name} for query: '{query_text}'")
            log_entry += f"\nAttempting search with {tool_to_use.name} for query: '{query_text}'."
            try:
                # Assuming search tools take a single 'query' string argument
                # The actual tool invocation might need specific formatting if it expects a dict
                tool_input = {"query": query_text}
                if tool_to_use.name == "tavily_search":  # Tavily specific params can be added if needed
                    tool_input["search_depth"] = "advanced"

                # Invoke the tool. `langchain_mcp_adapters` tools are LangChain BaseTools.
                result_content = await tool_to_use.ainvoke(tool_input)
                logger.debug(f"Raw result from {tool_to_use.name} for "
                             f"'{query_text}': {result_content}")

                # Store results. Assume result_content is a list of dicts or a dict with a 'results' key.
                current_query_results = []
                if isinstance(result_content,
                              dict) and 'results' in result_content:
                    current_query_results = result_content['results']
                elif isinstance(result_content, list):
                    current_query_results = result_content
                else:
                    logger.warning(
                        f"Unexpected result format from {tool_to_use.name}: {type(result_content)}"
                    )

                search_results_agg[
                    f"{tool_to_use.name}::{query_text}"] = current_query_results
                log_entry += (
                    f"\nSearch successful with {tool_to_use.name} for "
                    f"'{query_text}'. Results: {len(current_query_results)} items."
                )
                messages.append(
                    ToolMessage(
                        content=str(result_content)[:500] + "...",
                        tool_call_id=f"{tool_to_use.name}_{query_idx}"))

            except Exception as e:
                logger.error(
                    f"Error calling {tool_to_use.name} for query "
                    f"'{query_text}': {e}",
                    exc_info=True)
                log_entry += (f"\nError with {tool_to_use.name} for query "
                              f"'{query_text}': {e}")
                # Indicate error in content
                messages.append(
                    ToolMessage(
                        content=f"Error during tool call: {e}",
                        tool_call_id=f"{tool_to_use.name}_{query_idx}"))

        return {
            **state, "search_results": search_results_agg,
            "research_log": current_log + [log_entry],
            "messages": messages
        }

    async def _analyze_search_results(self,
                                      state: ResearchState) -> ResearchState:
        logger.info(f"Analyzing search results for: "
                    f"{state.get('company_name_to_research')}")
        log_entry = "Analyzing search results..."
        current_log = state.get("research_log", [])
        messages = state.get("messages", [])

        # Consolidate search results for the LLM to analyze
        formatted_search_results = ""
        for query, results in state.get("search_results", {}).items():
            formatted_search_results += f"Results for query '{query}':\n"
            if isinstance(results, list):
                for i, res in enumerate(
                        results[:3]):  # Show top 3 results per query
                    title = res.get('title', 'N/A')
                    url = res.get('url', 'N/A')
                    snippet = res.get('snippet', res.get(
                        'description', 'N/A'))  # Common keys for snippets
                    formatted_search_results += (
                        f"  {i+1}. Title: {title}\n     URL: {url}\n     "
                        f"Snippet: {snippet[:200]}...\n")
            else:
                formatted_search_results += f"  Unexpected format for query '{query}': {type(results)}\n"
            formatted_search_results += "\n"

        if not formatted_search_results:
            log_entry += " No search results to analyze."
            logger.info("No search results found to analyze.")
            # Decide whether to retry planning or end if no results
            messages.append(
                AIMessage(content=(
                    "No search results were found. Let's try planning "
                    "again or broadening the search.")))
            return {
                **state, "research_log": current_log + [log_entry],
                "messages": messages,
                "_next_step_decision": "plan_research"
            }

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             """You are a research analyst. Your task is to analyze the provided search results.
            Identify key URLs that seem most promising for detailed information extraction to build a company profile.
            Prioritize official company websites, reputable news articles, funding announcements, and in-depth profiles.
            Avoid forum links or very brief mentions unless they seem highly relevant.

            Based on the analysis, output a JSON object with:
            - 'urls_to_fetch': A list of unique URLs (max 10) to fetch for detailed content.
            - 'summary_of_findings': A brief text summary of key information gleaned from the search result snippets.
            - 'refine_search_queries': (Optional) A list of new search queries if current results are insufficient or need refinement.
            - 'proceed_to_extraction': (Optional) Boolean, true if you think enough context might be available from these snippets to attempt extraction, or if specific URLs are highly promising.
            """),
            ("human",
             "Company being researched: {company_name}\n\nSearch Results Snippets:\n{search_results_str}\n\nPreviously fetched URLs (if any):\n{fetched_urls_str}"
             )
        ])
        chain = prompt | self.llm
        response = await chain.ainvoke({
            "company_name":
            state.get("company_name_to_research", state['startup_query']),
            "search_results_str":
            formatted_search_results,
            "fetched_urls_str":
            "\n".join(state.get("fetched_content", {}).keys())
        })

        messages.append(response)
        analysis_data = {}  # Initialize
        try:
            content_to_parse = response.content
            if isinstance(response.content, str):
                json_start = content_to_parse.find('{')
                json_end = content_to_parse.rfind('}') + 1
                if json_start != -1 and json_end != -1:
                    content_to_parse = content_to_parse[json_start:json_end]
                analysis_data = json.loads(content_to_parse)
            elif isinstance(response.content, dict):
                analysis_data = response.content

            logger.info(f"Search results analysis: {analysis_data}")
            log_entry += f"\nAnalysis: {analysis_data.get('summary_of_findings', 'N/A')}"
            log_entry += f"\nIdentified URLs to fetch: {analysis_data.get('urls_to_fetch', [])}"

            updated_state = state.copy()
            updated_state["research_log"] = current_log + [log_entry]
            updated_state["messages"] = messages
            new_urls_to_fetch = list(
                set(
                    state.get("urls_to_fetch", []) +
                    analysis_data.get("urls_to_fetch", []))
            )  # Merge and deduplicate
            updated_state[
                "urls_to_fetch"] = new_urls_to_fetch[:
                                                     10]  # Limit to 10 URLs overall for fetching in one go

            # Ensure search_queries is a list
            search_queries_update = analysis_data.get("refine_search_queries",
                                                      [])
            if not isinstance(search_queries_update, list):
                logger.warning(
                    f"'refine_search_queries' from LLM was not a list: {search_queries_update}. Ignoring refinement."
                )
                search_queries_update = []

            updated_state["search_queries"] = search_queries_update
            updated_state[
                "_next_step_decision"] = "plan_research" if updated_state[
                    "search_queries"] else "fetch_content"

            return updated_state

        except json.JSONDecodeError as e:
            logger.error(
                f"Failed to parse search analysis JSON: {e}. Response: {response.content}"
            )
            log_entry += f"\nError parsing search analysis: {e}"
            messages.append(
                AIMessage(content=(f"Error: Could not parse search analysis. "
                                   f"Raw: {response.content}")))
            return {
                **state, "research_log": current_log + [log_entry],
                "messages": messages,
                "_next_step_decision": "handle_error"
            }
        except Exception as e:
            logger.error(f"Unexpected error in _analyze_search_results: {e}")
            log_entry += f"\nUnexpected error in search analysis: {e}"
            messages.append(
                AIMessage(content=(
                    f"Error: Unexpected error during search analysis. "
                    f"Raw: {response.content}")))
            return {
                **state, "research_log": current_log + [log_entry],
                "messages": messages,
                "_next_step_decision": "handle_error"
            }

    async def _fetch_content(self, state: ResearchState) -> ResearchState:
        logger.info(f"Fetching content for URLs: {state.get('urls_to_fetch')}")
        log_entry = "Fetching content..."
        current_log = state.get("research_log", [])
        messages = state.get("messages", [])
        fetched_content_agg = state.get("fetched_content", {})

        if not self.tools:
            log_entry += " Error: MCP tools not initialized."
            logger.error("MCP tools not initialized in _fetch_content")
            messages.append(
                AIMessage(content="Error: Fetch tools are not available."))
            return {
                **state, "research_log": current_log + [log_entry],
                "messages": messages,
                "_next_step_decision": "handle_error"
            }

        fetch_tool = next((t for t in self.tools if "fetch" in t.name.lower()),
                          None)
        if not fetch_tool:
            log_entry += " Error: Fetch tool not available."
            logger.error("Fetch tool not found.")
            messages.append(
                AIMessage(content="Error: Fetch tool is not available."))
            return {
                **state, "research_log": current_log + [log_entry],
                "messages": messages,
                "_next_step_decision": "handle_error"
            }

        urls_to_fetch_now = state.get("urls_to_fetch", [])
        for url_idx, url_to_fetch in enumerate(urls_to_fetch_now):
            if url_to_fetch in fetched_content_agg:  # Avoid re-fetching
                logger.info(f"Skipping already fetched URL: {url_to_fetch}")
                continue

            log_entry += f"\nAttempting to fetch: {url_to_fetch}"
            logger.info(
                f"Fetching URL: {url_to_fetch} using {fetch_tool.name}")
            try:
                # Fetch tool might expect {"url": "http://..."}
                tool_input = {"url": url_to_fetch}
                # Add other params if fetch_mcp supports them, e.g., 'output_format': 'markdown' or 'text'
                # result_content = await fetch_tool.ainvoke(tool_input)
                # The `fetch_mcp.py` from zcaceres/fetch-mcp seems to return a dict like {"content": "...", "url": "...", "status": ...}
                # Let's assume the main content is in a 'content' key.
                raw_fetch_result = await fetch_tool.ainvoke(tool_input)

                # Defensive coding for various possible fetch tool outputs
                content = ""
                if isinstance(raw_fetch_result, dict):
                    if "content" in raw_fetch_result:
                        content = raw_fetch_result["content"]
                    elif "markdown" in raw_fetch_result:  # common output format
                        content = raw_fetch_result["markdown"]
                    elif "text" in raw_fetch_result:
                        content = raw_fetch_result["text"]
                    else:
                        content = str(raw_fetch_result
                                      )  # fallback to string representation
                elif isinstance(raw_fetch_result, str):
                    content = raw_fetch_result
                else:
                    logger.warning(
                        f"Unexpected content type from fetch tool for {url_to_fetch}: {type(raw_fetch_result)}"
                    )
                    content = str(raw_fetch_result)  # fallback

                fetched_content_agg[
                    url_to_fetch] = content[:settings.
                                            MAX_FETCHED_CONTENT_LENGTH_PER_URL]  # Truncate if too long
                log_entry += (f"\nSuccessfully fetched: {url_to_fetch}. "
                              f"Content length (truncated): "
                              f"{len(fetched_content_agg[url_to_fetch])}")
                messages.append(
                    ToolMessage(
                        content=
                        f"Fetched {url_to_fetch} - content length: {len(content)}",
                        tool_call_id=f"fetch_{url_idx}"))
            except Exception as e:
                logger.error(f"Error fetching URL {url_to_fetch}: {e}",
                             exc_info=True)
                log_entry += f"\nError fetching {url_to_fetch}: {e}"
                fetched_content_agg[
                    url_to_fetch] = f"Error fetching content: {e}"  # Store error for context
                messages.append(
                    ToolMessage(content=f"Error fetching {url_to_fetch}: {e}",
                                tool_call_id=f"fetch_{url_idx}"))

        return {
            **state,
            "fetched_content": fetched_content_agg,
            "urls_to_fetch": [],  # Clear URLs that were attempted
            "research_log": current_log + [log_entry],
            "messages": messages
        }

    async def _extract_information(self,
                                   state: ResearchState) -> ResearchState:
        logger.info(f"Extracting information for: "
                    f"{state.get('company_name_to_research')}")
        log_entry = "Extracting information..."
        current_log = state.get("research_log", [])
        messages = state.get("messages", [])

        if not state.get("fetched_content") and not state.get(
                "search_results"):
            log_entry += " No content or search snippets to extract from."
            logger.warning("No content available for extraction.")
            messages.append(
                AIMessage(
                    content="No content to extract from. Planning again."))
            return {
                **state, "research_log": current_log + [log_entry],
                "messages": messages,
                "_next_step_decision": "plan_research"
            }

        # Consolidate content for LLM. Prioritize fetched full content.
        # Also include search snippets for broader context if full content is sparse.
        extraction_context = ""
        if state.get("fetched_content"):
            for url, content in state["fetched_content"].items():
                extraction_context += f"Content from {url}:\n{content}\n\n---\n\n"

        if not extraction_context and state.get("search_results"):
            logger.info(
                "No full fetched content, using search snippets for extraction attempt."
            )
            log_entry += " No full fetched content, attempting extraction from search snippets."
            for query, results in state.get("search_results", {}).items():
                if isinstance(results, list):
                    for res in results:
                        snippet = res.get('snippet', res.get('description'))
                        if snippet:
                            extraction_context += f"Search snippet (query: {query}): {snippet}\n"
            extraction_context += "\n---\n\n"

        if not extraction_context:
            log_entry += " No context available for extraction even after checking snippets."
            logger.warning(
                "No context (neither fetched nor snippets) for extraction.")
            messages.append(
                AIMessage(
                    content=
                    "No context available for extraction. Planning again."))
            return {
                **state, "research_log": current_log + [log_entry],
                "messages": messages,
                "_next_step_decision": "plan_research"
            }

        # Using Pydantic model for structured output directly if LLM supports it well (e.g. OpenAI with `with_structured_output`)
        # Otherwise, ask for JSON and parse into Pydantic model.
        # For OpenAI, we can use .with_structured_output(CompanyPydantic) if this feature is stable and works well.
        # Let's try asking for JSON that conforms to the Pydantic schema first.

        company_schema_description = CompanyPydantic.schema_json(indent=2)

        prompt_messages = [(
            "system",
            f"""You are a data extraction specialist. Your task is to extract detailed information 
            about a company from the provided text and structure it according to the given JSON schema. 
            The company being researched is: {state.get('company_name_to_research', state['startup_query'])}. 
            Fill in as much information as possible. If some information is not available, omit the fields or set them to null where appropriate.
            Pay close attention to nested structures like founders, products, funding_rounds, etc.
            Ensure dates are in YYYY-MM-DD format if possible.
            
            Target JSON Schema to populate for the company:
            {company_schema_str}
            
            Respond ONLY with the JSON object that conforms to this schema. Do not add any explanatory text before or after the JSON.
            """), ("human", "Context to extract from:\n\n{context_str}")]
        extraction_prompt = ChatPromptTemplate.from_messages(prompt_messages)

        # Bind the Pydantic model for structured output (preferred method for OpenAI)
        structured_llm = self.llm.with_structured_output(CompanyPydantic)
        extraction_chain = extraction_prompt | structured_llm

        max_context_len = getattr(settings,
                                  'MAX_CONTEXT_LENGTH_FOR_EXTRACTION', 100000)
        try:
            extracted_data_pydantic = await extraction_chain.ainvoke({
                "company_name_to_research":
                state.get("company_name_to_research", state['startup_query']),
                "company_schema_str":
                company_schema_description,  # For LLM's context if needed
                "context_str":
                extraction_context[:max_context_len]  # Truncate context
            })

            logger.info(f"Successfully extracted data into Pydantic model: "
                        f"{type(extracted_data_pydantic)}")
            log_entry += (
                f"\nSuccessfully extracted data. Company name: "
                f"{extracted_data_pydantic.name if extracted_data_pydantic else 'N/A'}."
            )
            messages.append(
                AIMessage(content=(
                    f"Extracted data for "
                    f"{extracted_data_pydantic.name if extracted_data_pydantic else 'company'}. "
                    f"Proceeding to save.")))
            return {
                **state,
                "extracted_company_data":
                extracted_data_pydantic,
                "extracted_raw_data":
                extracted_data_pydantic.dict()
                if extracted_data_pydantic else None,  # Store raw dict as well
                "research_log":
                current_log + [log_entry],
                "messages":
                messages,
                "_next_step_decision":
                "validate_and_save_data"
            }

        except Exception as e:  # Catch errors from LLM structured output or validation
            logger.error(f"Error during structured data extraction: {e}",
                         exc_info=True)
            log_entry += f"\nError extracting structured data: {e}"
            # Fallback: try to get JSON and parse manually if structured_output failed
            try:
                logger.info(
                    "Falling back to non-structured JSON extraction due to error."
                )
                fallback_chain = extraction_prompt | self.llm  # Plain LLM call, expecting JSON string
                response_json_str = await fallback_chain.ainvoke({
                    "company_name_to_research":
                    state.get("company_name_to_research",
                              state['startup_query']),
                    "company_schema_str":
                    company_schema_description,
                    "context_str":
                    extraction_context[:max_context_len]
                })
                json_text = response_json_str.content
                json_start_fb = json_text.find('{')
                json_end_fb = json_text.rfind('}') + 1
                if json_start_fb != -1 and json_end_fb != -1:
                    json_text = json_text[json_start_fb:json_end_fb]

                parsed_raw_data = json.loads(json_text)
                validated_data = CompanyPydantic(**parsed_raw_data)
                logger.info(
                    "Fallback JSON extraction and Pydantic validation successful."
                )
                log_entry += "\nFallback JSON extraction successful."
                messages.append(
                    AIMessage(
                        content=
                        f"Extracted data via fallback for {validated_data.name}. Proceeding to save."
                    ))
                return {
                    **state, "extracted_company_data": validated_data,
                    "extracted_raw_data": parsed_raw_data,
                    "research_log": current_log + [log_entry],
                    "messages": messages,
                    "_next_step_decision": "validate_and_save_data"
                }
            except Exception as fallback_e:
                logger.error(
                    f"Fallback JSON extraction also failed: {fallback_e}",
                    exc_info=True)
                log_entry += f"\nFallback JSON extraction also failed: {fallback_e}"
                messages.append(
                    AIMessage(
                        content=f"Error: Data extraction failed. {fallback_e}")
                )
                return {
                    **state, "research_log": current_log + [log_entry],
                    "messages": messages,
                    "_next_step_decision": "handle_error"
                }

    def _validate_and_save_data(self, state: ResearchState) -> ResearchState:
        logger.info(f"Validating and saving data for: "
                    f"{state.get('company_name_to_research')}")
        log_entry = "Validating and saving data..."
        current_log = state.get("research_log", [])
        messages = state.get("messages", [])
        extracted_data = state.get("extracted_company_data")

        if not extracted_data:
            log_entry += " No extracted data to save."
            logger.warning("No data to validate and save.")
            messages.append(
                AIMessage(content="No data was extracted. Cannot save."))
            return {
                **state, "research_log": current_log + [log_entry],
                "messages": messages,
                "_next_step_decision": "plan_research"
            }  # Or handle_error

        if not isinstance(extracted_data, CompanyPydantic):
            try:
                # If it's a dict (e.g. from raw_extraction), try to parse
                validated_data = CompanyPydantic(**extracted_data)
                logger.info(
                    "Successfully validated raw dict into CompanyPydantic model."
                )
                extracted_data = validated_data
            except Exception as e:
                log_entry += f" Validation failed for extracted data: {e}"
                logger.error(f"Validation of extracted data failed: {e}. "
                             f"Data: {extracted_data}")
                messages.append(
                    AIMessage(
                        content=f"Error: Extracted data failed validation. {e}"
                    ))
                return {
                    **state, "research_log": current_log + [log_entry],
                    "messages": messages,
                    "_next_step_decision": "handle_error"
                }

        # Ensure company name in extracted data matches (or is set if missing)
        if not extracted_data.name:
            target_company_name = state.get("company_name_to_research",
                                            state['startup_query'])
            logger.warning(
                f"Extracted company data missing name. Setting to target: "
                f"{target_company_name}")
            extracted_data.name = target_company_name
            log_entry += f" Set company name to '{target_company_name}' as it was missing in extraction."

        try:
            saved_company_orm = upsert_company_data(extracted_data)
            if saved_company_orm and saved_company_orm.company_id is not None:
                log_entry += f" Data successfully saved/updated for company ID: {saved_company_orm.company_id}."
                logger.info(
                    f"Data saved for company {saved_company_orm.name} (ID: {saved_company_orm.company_id})"
                )
                messages.append(
                    AIMessage(
                        content=
                        f"Successfully saved data for {saved_company_orm.name}."
                    ))
                return {
                    **state, "research_log": current_log + [log_entry],
                    "saved_company_id": saved_company_orm.company_id,
                    "messages": messages
                }
            else:
                log_entry += " Failed to save data to database (upsert returned None or no ID)."
                logger.error(f"Failed to save data for {extracted_data.name}. "
                             f"upsert_company_data returned None or no ID.")
                messages.append(
                    AIMessage(
                        content=
                        f"Error: Could not save data for {extracted_data.name} to database."
                    ))
                return {
                    **state, "research_log": current_log + [log_entry],
                    "messages": messages,
                    "_next_step_decision": "handle_error"
                }
        except Exception as e:
            log_entry += f" Error during database save operation: {e}"
            logger.error(f"Database save error for {extracted_data.name}: {e}",
                         exc_info=True)
            messages.append(
                AIMessage(content=f"Error: Database operation failed. {e}"))
            return {
                **state, "research_log": current_log + [log_entry],
                "messages": messages,
                "_next_step_decision": "handle_error"
            }

    def _handle_error(self, state: ResearchState) -> ResearchState:
        logger.error(
            f"Research process encountered an error for: "
            f"{state.get('company_name_to_research', state['startup_query'])}")
        log_entry = "Research process terminated due to an error."
        # Log the last few messages for context on the error
        last_messages = "\n".join([
            f"{m.type}: {str(m.content)[:200]}..."
            for m in state.get("messages", [])[-3:]
        ])
        log_entry += f"\nFinal messages leading to error:\n{last_messages}"
        current_log = state.get("research_log", [])
        final_log = current_log + [log_entry]
        logger.info(f"Final research log for {state.get('startup_query')}:\n" +
                    "\n".join(final_log))
        # Potentially save error state to DB or a file
        return {**state, "research_log": final_log}  # Ends the graph here

    # --- Conditional Edges Logic ---
    def _decide_after_planning(self, state: ResearchState) -> str:
        decision = state.get("_next_step_decision", "handle_error")
        logger.debug(f"Decision after planning: {decision}")
        return decision

    def _decide_after_search_analysis(self, state: ResearchState) -> str:
        decision = state.get("_next_step_decision", "handle_error")
        logger.debug(f"Decision after search analysis: {decision}")
        return decision

    def _decide_after_extraction(self, state: ResearchState) -> str:
        decision = state.get("_next_step_decision", "handle_error")
        logger.debug(f"Decision after extraction: {decision}")
        # Ensure we only go to save if data exists and decision is correct
        if decision == "validate_and_save_data" and not state.get(
                "extracted_company_data"):
            logger.warning(
                "Decision was to save, but no extracted data found. Handling error."
            )
            return "handle_error"
        return decision

    # --- Main Execution Method ---
    async def run_research_for_startup(self,
                                       startup_query: str) -> ResearchState:
        logger.info(f"Starting research for startup query: {startup_query}")
        await self._setup_mcp_client_and_tools()  # Ensure client is ready

        initial_state: ResearchState = {
            "startup_query":
            startup_query,
            "company_name_to_research":
            None,
            "search_queries": [],
            "search_results": {},
            "urls_to_fetch": [],
            "fetched_content": {},
            "extracted_raw_data":
            None,
            "extracted_company_data":
            None,
            "research_log": [f"Initiating research for: {startup_query}"],
            "messages": [
                HumanMessage(
                    content=f"Research the following startup: {startup_query}")
            ],
            "saved_company_id":
            None,
            "_next_step_decision":
            None  # No initial decision needed
        }

        config = {"recursion_limit": settings.LANGGRAPH_RECURSION_LIMIT}
        # Can add "configurable": {"thread_id": "some_unique_id_for_startup_query"} for persistence with appropriate checkpointer

        final_state = await self.graph.ainvoke(initial_state, config=config)
        logger.info(
            f"Research finished for '{startup_query}'. Final state: {final_state.get('saved_company_id') is not None}"
        )
        logger.info(f"Full research log for '{startup_query}':\n" +
                    "\n".join(final_state.get("research_log", [])))
        return final_state


async def main():
    # Initialize database (create tables if they don't exist)
    # This should ideally be run once at application startup, not per agent run.
    try:
        init_db()
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}. Exiting.")
        return

    # Load startups from startups.yaml
    try:
        with open("startups.yaml", 'r') as f:
            startup_data = yaml.safe_load(f)
        startup_queries = startup_data.get("startups", [])
        if not startup_queries:
            logger.warning("No startups found in startups.yaml")
            return
    except FileNotFoundError:
        logger.error(
            "startups.yaml not found. Please create it with a list of startups."
        )
        return
    except yaml.YAMLError as e:
        logger.error(f"Error parsing startups.yaml: {e}")
        return

    agent = ResearchAgent()
    for query in startup_queries:
        if isinstance(query, str) and query.strip():
            logger.info(f"--- Processing startup: {query} ---")
            await agent.run_research_for_startup(query)
            logger.info(f"--- Finished processing startup: {query} ---")
        else:
            logger.warning(f"Skipping invalid startup entry: {query}")

    # Important: Clean up MCP client resources
    if agent.mcp_client:
        # Changed from stop_servers to stop()
        await agent.mcp_client.stop()
        logger.info("MCP servers stopped.")


if __name__ == "__main__":
    # Setup OPENAI_API_KEY and other necessary env vars (TAVILY_API_KEY, BRAVE_API_KEY if MCP servers need them)
    # Also, ensure MCP_SERVER_PATH is correctly set in your .env or config.py
    # e.g., os.environ["OPENAI_API_KEY"] = "sk-..."
    # settings.OPENAI_API_KEY = "sk-..." # if using pydantic settings
    asyncio.run(main())
    # e.g., os.environ["OPENAI_API_KEY"] = "sk-..."
    # settings.OPENAI_API_KEY = "sk-..." # if using pydantic settings
    asyncio.run(main())
