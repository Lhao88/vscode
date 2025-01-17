# from peft import AutoPeftModelForCausalLM
# from transformers import AutoTokenizer
#
# # model_dir = "E:\pythonProject\Qwen\\finetune\output_qwen"
# model_dir = "E:\my_model\qwen_1.8B_lora"
# model = AutoPeftModelForCausalLM.from_pretrained(
#     f'{model_dir}',
#     device_map="auto",
#     trust_remote_code=True
# ).eval()
#
# tokenizer = AutoTokenizer.from_pretrained(f'{model_dir}', revision='master', trust_remote_code=True)
#
# response, history = model.chat(tokenizer, "电力市场与普通商品市场有哪些差异？", history=None)
# print(response)

# 加载qwen-1.8B_lora 模型
from modelscope import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# Note: The default behavior now has injection attack prevention off.

tokenizer = AutoTokenizer.from_pretrained("E:\my_model\qwen-1_8B-Chat_lora",  trust_remote_code=True)

# use auto mode, automatically select precision based on the device.
model = AutoModelForCausalLM.from_pretrained("E:\my_model\qwen-1_8B-Chat_lora", device_map="auto", trust_remote_code=True).eval()
# Specify hyperparameters for generation. But if you use transformers>=4.32.0, there is no need to do this.
# model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-1_8B-Chat", trust_remote_code=True) # 可指定不同的生成长度、top_p等相关超参

# 第一轮对话 1st dialogue turn
response, history = model.chat(tokenizer, "请问电力市场有哪些特征？", history=None)
print(response)
# 你好！很高兴为你提供帮助。