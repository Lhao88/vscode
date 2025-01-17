# import os
# os.environ["HF_DATASETS_CACHE"] = "/disk/cache/"
# os.environ["HF_HOME"] = "/disk/cache/"
# os.environ["HUGGINGFACE_HUB_CACHE"] = "/disk/cache/"
# os.environ["TRANSFORMERS_CACHE"] = "/disk/cache/"

from transformers import AutoTokenizer, AutoModel
# tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b-int4", trust_remote_code=True)
# model = AutoModel.from_pretrained("THUDM/chatglm-6b-int4", trust_remote_code=True).half().cuda()
# tokenizer = AutoTokenizer.from_pretrained("E:\huggingface\hub\models--THUDM--chatglm-6b-int4\snapshots\826ca34b74d484f40448238e57a0b45b66ad30fb", trust_remote_code=True)
# model = AutoModel.from_pretrained("E:\huggingface\hub\models--THUDM--chatglm-6b-int4\snapshots\826ca34b74d484f40448238e57a0b45b66ad30fb", trust_remote_code=True).half().cuda()
# response, history = model.chat(tokenizer, "你好", history=[])
# print(response)
# response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
# print(response)

from PowerMarketQA.CustomLLM import CustomLLM

llm = CustomLLM(n=10)
print(llm("你好！"))
