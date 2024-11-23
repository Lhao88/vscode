from typing import Any, List, Mapping, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM

# 集成`LLM`
class CustomLLM(LLM):
    n: int

    @property
    def _llm_type(self) -> str:
        # 返回我们自定义的模型标记
        return "custom_llm"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        # 在这里实现模型api调用
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        return prompt[: self.n]

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        # 可选debug信息
        """Get the identifying parameters."""
        return {"n": self.n}