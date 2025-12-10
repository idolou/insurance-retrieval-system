from typing import Any, Callable, List, Optional, Union

from llama_index.core.agent import AgentWorkflow
from llama_index.core.tools import BaseTool


class SimpleAgent:
    def __init__(
        self,
        tools: List[Union[BaseTool, Callable]],
        llm: Optional[Any],
        system_prompt: str,
    ) -> None:
        # Use the modern AgentWorkflow which automatically selects ReAct or FunctionCalling
        self.workflow = AgentWorkflow.from_tools_or_functions(
            tools_or_functions=tools, llm=llm, system_prompt=system_prompt, verbose=True
        )

    async def achat(self, user_query: str) -> Any:
        # Run the workflow asynchronously
        # run() returns a handler that we can await to get the final result
        handler = self.workflow.run(user_msg=user_query)
        response = await handler
        return response

    def chat(self, user_query: str) -> Any:
        # Synchronous wrapper for the async workflow
        import asyncio

        # Simply run the async method
        return asyncio.run(self.achat(user_query))
