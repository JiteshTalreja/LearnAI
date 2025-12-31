from typing import Any, Dict
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


class AgentCallbackHandler(BaseCallbackHandler):
    def on_llm_start(
    self, serialized: Dict[str, Any], prompts: list[str], **kwargs: Any
    ) -> Any:
        "Run when llm starts"
        print(f"***Prompts sent to LLM***\n{prompts[0]}")
        print("**************************")

    def on_llm_end(
        self,
        response: LLMResult,
        **kwargs: Any,
    ) -> Any:
        "Run when llm ends"
        print(f"***LLM Response***\n{response.generations[0][0].text}")
        print("*******************")