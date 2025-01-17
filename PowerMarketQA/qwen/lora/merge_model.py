from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

path_to_adapter = "E:\pythonProject\Qwen\\finetune\output_qwen"

model = AutoPeftModelForCausalLM.from_pretrained(
    path_to_adapter, # path to the output directory
    device_map="auto",
    trust_remote_code=True
).eval()


new_model_directory = "E:\my_model\qwen-1_8B-Chat_lora"
merged_model = model.merge_and_unload()
# max_shard_size and safe serialization are not necessary.
# They respectively work for sharding checkpoint and save the model to safetensors
merged_model.save_pretrained(new_model_directory)


tokenizer = AutoTokenizer.from_pretrained(
    path_to_adapter, # path to the output directory
    trust_remote_code=True
)
tokenizer.save_pretrained(new_model_directory)