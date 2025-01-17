from modelscope import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# Note: The default behavior now has injection attack prevention off.

tokenizer = AutoTokenizer.from_pretrained("E:\modelspace\hub\Qwen\Qwen-1_8B-Chat",  trust_remote_code=True)

# use auto mode, automatically select precision based on the device.
model = AutoModelForCausalLM.from_pretrained("E:\modelspace\hub\Qwen\Qwen-1_8B-Chat", device_map="auto", trust_remote_code=True).eval()
# Specify hyperparameters for generation. But if you use transformers>=4.32.0, there is no need to do this.
# model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-1_8B-Chat", trust_remote_code=True) # 可指定不同的生成长度、top_p等相关超参

# 第一轮对话 1st dialogue turn
response, history = model.chat(tokenizer, "你好", history=None)
print(response)
# 你好！很高兴为你提供帮助。

