# import the libraries
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig
from datasets import load_dataset
from trl import SFTTrainer
import torch

# model
model_name = "mistralai/Mistral-7B-v0.1"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# LoRa
model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_config)
model.gradient_checkpointing_disable()

# dataset
dataset = load_dataset("json", data_files="data.jsonl")["train"]

def format_example(example):
    return {"text": f"<s>[INST] {example['instruction']} [/INST] {example['output']}</s>"}

dataset = dataset.map(format_example)

# training
training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    max_steps=100,
    output_dir="./results",
    fp16=False,
    bf16=False
)

def formatting_func(example):
    return example["text"]

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    formatting_func=formatting_func,
    args=training_args,
)

# train the model
trainer.train()

# save the model
trainer.model.save_pretrained("final-lora")
tokenizer.save_pretrained("final-lora")

# check 
import os
print(os.listdir("final-lora"))

# login to Hugging Face Hub
from huggingface_hub import login
login()

# check the files to be pushed
import os
print(os.listdir())

# check the lora files
print(os.listdir("final-lora"))

from huggingface_hub import login
login()

# push the model to Hugging Face Hub
from huggingface_hub import HfApi, upload_folder

api = HfApi()

api.create_repo("Senaa2402/mistral-qlora", exist_ok=True)

upload_folder(
    folder_path="final-lora",
    repo_id="Senaa2402/mistral-qlora",
    repo_type="model"
)

# upload
from huggingface_hub import upload_folder, login

login()

upload_folder(
    folder_path="final-lora",
    repo_id="Senaa2402/mistral-qlora",
    repo_type="model"
)

# model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

model_name = "mistralai/Mistral-7B-v0.1"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

model = PeftModel.from_pretrained(base_model, "Senaa2402/mistral-qlora")
tokenizer = AutoTokenizer.from_pretrained("Senaa2402/mistral-qlora")

# model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

model_name = "mistralai/Mistral-7B-v0.1"
adapter_name = "Senaa2402/mistral-qlora"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

model = PeftModel.from_pretrained(base_model, adapter_name)
tokenizer = AutoTokenizer.from_pretrained(adapter_name)

# test
def chat(prompt):
    inputs = tokenizer(f"<s>[INST] {prompt} [/INST]", return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

print(chat("What is AI?"))

#check with gradio UI
!pip install gradio
import gradio as gr

def chat_ui(prompt):
    inputs = tokenizer(f"<s>[INST] {prompt} [/INST]", return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

gr.Interface(fn=chat_ui, inputs="text", outputs="text").launch()