from typing import Any, List, Mapping, Optional
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from transformers import AutoTokenizer, AutoModel
# 集成`LLM`
class CustomLLM(LLM):
    n:int

    @property
    def _llm_type(self) -> str:
        # 返回我们自定义的模型标记
        return "custom_llm_chat-glm-6b-int4"

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

        tokenizer = AutoTokenizer.from_pretrained(
            "E:\huggingface\hub\models--THUDM--chatglm-6b-int4\snapshots\826ca34b74d484f40448238e57a0b45b66ad30fb",
            trust_remote_code=True)
        model = AutoModel.from_pretrained(
            "E:\huggingface\hub\models--THUDM--chatglm-6b-int4\snapshots\826ca34b74d484f40448238e57a0b45b66ad30fb",
            trust_remote_code=True).half().cuda()
        response, history = model.chat(tokenizer, prompt, history=[])

        return response

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        # 可选debug信息
        """Get the identifying parameters."""
        return {"n": self.n}