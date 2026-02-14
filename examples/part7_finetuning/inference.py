"""
Part 7: Loading and Using the Fine-Tuned Model
================================================
Loads the base model + LoRA adapter and generates text.

Run finetune_lora.py first to create the adapter.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load base model + LoRA adapter
base_model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = PeftModel.from_pretrained(base_model, "./lora_adapter")
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Generate
prompt = "<|user|>\nWhat is attention in transformers?\n<|assistant|>\n"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.7, do_sample=True)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
