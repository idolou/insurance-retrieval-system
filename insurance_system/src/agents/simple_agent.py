import asyncio
from typing import Any, Callable, List, Optional, Union

from llama_index.core.agent import AgentWorkflow
from llama_index.core.tools import BaseTool


class SimpleAgentError(Exception):
    """Base exception for SimpleAgent errors."""

    pass


class SimpleAgent:
    def __init__(
        self,
        tools: List[Union[BaseTool, Callable]],
        llm: Optional[Any],
        system_prompt: str,
    ) -> None:
        """
        Initialize the Simple Agent.

        Args:
            tools: List of tools or callable functions for the agent.
            llm: Language model instance.
            system_prompt: System prompt for the agent.

        Raises:
            SimpleAgentError: If agent initialization fails.
        """
        try:
            # Use the modern AgentWorkflow which automatically selects ReAct or FunctionCalling
            self.workflow = AgentWorkflow.from_tools_or_functions(
                tools_or_functions=tools,
                llm=llm,
                system_prompt=system_prompt,
                verbose=True,
            )
        except Exception as e:
            raise SimpleAgentError(f"Agent initialization failed: {e}") from e

    async def achat(self, user_query: str) -> Any:
        """
        Asynchronously chat with the agent.

        Args:
            user_query: The user's query string.

        Returns:
            Agent response object.

        Raises:
            SimpleAgentError: If chat execution fails.
        """
        if not user_query or not user_query.strip():
            raise ValueError("User query cannot be empty")

        try:
            # Run the workflow asynchronously
            # run() returns a handler that we can await to get the final result
            handler = self.workflow.run(user_msg=user_query)
            response = await handler
            return response
        except ValueError:
            raise
        except Exception as e:
            raise SimpleAgentError(f"Chat execution failed: {e}") from e

    def chat(self, user_query: str) -> Any:
        """
        Synchronously chat with the agent.

        This method creates a new event loop for synchronous execution.
        If called from an async context, use achat() instead.

        Args:
            user_query: The user's query string.

        Returns:
            Agent response object.

        Raises:
            SimpleAgentError: If chat execution fails.
        """
        if not user_query or not user_query.strip():
            raise ValueError("User query cannot be empty")

        try:
            # Check if we're in an async context
            try:
                asyncio.get_running_loop()
                # We're in an async context - can't use asyncio.run()
                error_msg = (
                    "Cannot use sync chat() in async context. Use achat() instead."
                )
                raise RuntimeError(error_msg)
            except RuntimeError as e:
                # Check if this is our intentional error (async context detected)
                if "Cannot use sync chat()" in str(e):
                    raise  # Re-raise our intentional error
                # Otherwise, this is the natural RuntimeError from get_running_loop()
                # when there's no running loop - proceed with sync execution
                pass

            # No running loop - this is the expected case for sync execution
            # Create new event loop for synchronous execution
            try:
                return asyncio.run(self.achat(user_query))
            except RuntimeError as e:
                if "no running event loop" in str(e).lower():
                    # Try alternative approach: create new loop manually
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        return loop.run_until_complete(self.achat(user_query))
                    finally:
                        loop.close()
                else:
                    raise
        except ValueError:
            raise
        except RuntimeError as e:
            if "Cannot use sync chat()" in str(e):
                raise  # Re-raise our intentional error
            raise SimpleAgentError(f"Chat execution failed: {e}") from e
        except Exception as e:
            raise SimpleAgentError(f"Chat execution failed: {e}") from e
