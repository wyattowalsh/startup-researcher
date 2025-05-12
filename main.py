import asyncio
import time

# Import the actual ResearchAgent
from startup_researcher.agents.research_agent import ResearchAgent
from startup_researcher.config import get_settings
from startup_researcher.logging import get_logger

# Remove the placeholder comment/import if it existed

logger = get_logger()

# Placeholder function 'run_research_for_startup' is no longer needed and will be removed.

async def main():
    """
    Main orchestration function. Reads startup list from config,
    instantiates the research agent, and runs research concurrently.
    """
    logger.info("Starting startup research process...")
    settings = get_settings()
    startups = settings.startups

    if not startups or startups == ["startup_1", "startup_2", "..."]:
        logger.warning(
            "No startups found in config.yaml or using default placeholder values. "
            "Please update 'startups.yaml' or 'config.yaml' with actual startup identifiers."
        )
        return

    logger.info(f"Found {len(startups)} startups to research: {startups}")

    # Instantiate the ResearchAgent
    # The agent will handle its own MCP client setup internally when run
    agent = ResearchAgent()
    agent_initialized = False # Flag to track if agent setup succeeded

    try:
        # Filter valid startup identifiers
        valid_startups = [
            s for s in startups
            if isinstance(s, str) and s not in ["...", "startup_1", "startup_2"]
        ]

        if not valid_startups:
            logger.warning("No valid startup identifiers to process after filtering.")
            return

        # Create tasks calling the agent's method directly
        logger.info(f"Dispatching research tasks for {len(valid_startups)} startups concurrently...")
        tasks = [
            agent.run_research_for_startup(startup) for startup in valid_startups
        ]

        # Run tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Log results/errors
        for i, result in enumerate(results):
            startup_name = valid_startups[i]
            is_exception = isinstance(result, Exception)
            is_successful_save = False
            is_error_state_from_agent = False

            if isinstance(result, dict):
                saved_id = result.get("saved_company_id")
                is_successful_save = saved_id is not None
                has_handle_error_log = "handle_error" in result.get("research_log", [""])[-1]
                # Considered an error state if saving failed AND the agent logged a handle_error step
                is_error_state_from_agent = (not is_successful_save) and has_handle_error_log
            elif result is None:
                # Treat None result as an error state
                is_error_state_from_agent = True
            
            # Decide logging based on flags
            if is_exception:
                logger.error(f"Research task for '{startup_name}' failed with exception: {result}", exc_info=result)
            elif is_error_state_from_agent:
                logger.warning(f"Research task for '{startup_name}' completed with an error state (check agent logs).")
            elif is_successful_save:
                logger.success(f"Research task for '{startup_name}' completed successfully. Saved ID: {saved_id}")
            else:
                # Catch-all for unexpected scenarios (e.g., dict result without saved_id but no handle_error log)
                logger.warning(f"Research task for '{startup_name}' completed with unexpected state: {str(result)[:200]}...")

        agent_initialized = agent.mcp_client is not None # Check if client was set up

    except Exception as e:
         logger.error(f"An error occurred during the main orchestration: {e}", exc_info=True)
    finally:
        # Clean up the agent's resources (like MCP client) if it was initialized
        if agent_initialized and agent.mcp_client:
            logger.info("Shutting down agent's MCP client...")
            try:
                await agent.mcp_client.stop_servers()
                logger.info("Agent's MCP client stopped successfully.")
            except Exception as cleanup_e:
                logger.error(f"Error stopping agent's MCP client: {cleanup_e}", exc_info=True)
        else:
            logger.info("Agent or its MCP client was not initialized, skipping cleanup.")

        logger.info("Startup research process finished.")


if __name__ == "__main__":
    # Corrected line: only one call
    asyncio.run(main())     asyncio.run(main()) 