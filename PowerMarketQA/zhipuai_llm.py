import zhipuai
from typing import Any
from llama_index.core.llms import CustomLLM,LLMMetadata,CompletionResponse
from llama_index.core.llms.callbacks import llm_completion_callback
class ZhipuLLM(CustomLLM):
    model_name: str = "chatglm_turbo"
    context_window: int = 3900
    num_output: int = 256

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
        )
    

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        response = invoke_prompt(prompt)
        return CompletionResponse(text=response)