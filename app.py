import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

model_name = "mistralai/Mistral-7B-v0.1"
adapter_name = "Senaa2402/mistral-qlora"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

# Load adapter
model = PeftModel.from_pretrained(base_model, adapter_name)
tokenizer = AutoTokenizer.from_pretrained(adapter_name)

def chat(prompt):
    inputs = tokenizer(f"<s>[INST] {prompt} [/INST]", return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

gr.Interface(fn=chat, inputs="text", outputs="text").launch()