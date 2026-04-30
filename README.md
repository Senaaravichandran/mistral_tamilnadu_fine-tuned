# 🚀 Mistral-7B Fine-Tuned Chatbot (Tamil Nadu Culture)

### QLoRA + Google Colab (T4) + Hugging Face + Gradio

---

## 🔗 Repository

👉 https://github.com/Senaaravichandran/mistral_tamilnadu_fine-tuned

## 🤖 Model (Hugging Face)

👉 https://huggingface.co/Senaa2407/mistral-qlora

---

# 📌 Project Overview

This project demonstrates **end-to-end fine-tuning of a Large Language Model (LLM)** using:

* **Mistral-7B**
* **QLoRA (4-bit Quantization + LoRA)**
* **Google Colab (Free T4 GPU)**

The model is trained on a **custom dataset (~500 Q&A pairs)** focused on **Tamil Nadu culture**, and deployed as an **interactive chatbot using Gradio**.

---

# 🎓 Learning & Certification Context

This project was developed based on concepts from:

### 🏆 DeepLearning.AI — Finetuning Large Language Models (Free Certification)

### Topics Applied:

* LoRA (Low-Rank Adaptation)
* QLoRA (Quantized LoRA)
* Supervised Fine-Tuning (SFT)
* Instruction-based training
* Efficient LLM training on low-resource GPUs

---

# 🧠 Skills Demonstrated

* Fine-tuning LLMs with QLoRA
* Memory-efficient training (4-bit quantization)
* Hugging Face Transformers ecosystem
* Dataset design (instruction → response format)
* Model deployment (Hugging Face Hub)
* Building chatbot UI using Gradio
* End-to-end ML pipeline (train → save → deploy → UI)

---

# 🏗️ Tech Stack

* Python
* Transformers
* PEFT (LoRA)
* BitsAndBytes (4-bit quantization)
* TRL (SFTTrainer)
* Hugging Face Hub
* Gradio (UI)
* Google Colab

---

# 📂 Project Structure

```id="v3tx7m"
mistral_tamilnadu_fine-tuned/
│
├── train.py              # Fine-tuning script
├── app.py                # Gradio chatbot UI
├── requirements.txt      # Dependencies
├── README.md             # Documentation
└── data/
    └── data.jsonl        # Custom dataset
```

---

# 📊 Dataset

### 🔹 Format

```json id="uv6l6v"
{"instruction": "What is Pongal?", "output": "Pongal is a harvest festival celebrated in Tamil Nadu."}
```

---

### 🔹 Why Tamil Nadu Dataset?

* 🇮🇳 Focus on **regional knowledge (India-specific)**
* 🎯 Demonstrates **domain-specific fine-tuning**
* 🧠 Improves contextual accuracy
* 📚 Clean Q&A structure → ideal for SFT

---

### 🔹 Dataset Features

* ~500 instruction-response pairs
* Consistent formatting
* Cultural, historical, and factual knowledge

---

# ⚙️ How to Run (Google Colab — Step by Step)

---

## 🔹 Step 1: Open Colab

👉 https://colab.research.google.com/

---

## 🔹 Step 2: Set Runtime

```id="x9gkbb"
Runtime → Change runtime type → GPU → Select T4
```

---

## 🔹 Step 3: Clone Repository

```python id="f8q07k"
!git clone https://github.com/Senaaravichandran/mistral_tamilnadu_fine-tuned.git
%cd mistral_tamilnadu_fine-tuned
```

---

## 🔹 Step 4: Install Dependencies

```python id="v2kx8j"
!pip install -r requirements.txt
```

---

## 🔹 Step 5: Train Model (QLoRA)

```python id="6s7g1v"
!python train.py
```

### What happens here:

* Loads Mistral-7B in 4-bit
* Applies LoRA adapters
* Fine-tunes on dataset
* Saves adapter → `final-lora/`

---

## 🔹 Step 6: Push to Hugging Face

```python id="5zx1p4"
from huggingface_hub import login
login()
```

---

## 🔹 Step 7: Run Chatbot (Gradio UI)

```python id="z1p9b8"
!python app.py
```

👉 This launches a **live chatbot interface**

---

# 💬 Chatbot Example

### Input:

```id="a7zz7r"
What is Bharatanatyam?
```

### Output:

```id="c0b1z9"
Bharatanatyam is a classical dance form originating from Tamil Nadu.
```

---

# 🎯 Key Features

* ✅ Fine-tuned Mistral-7B using QLoRA
* ✅ Runs on free GPU (Colab T4)
* ✅ Lightweight adapter-based training
* ✅ Hosted model on Hugging Face
* ✅ Interactive chatbot with Gradio

---

# ⚠️ Limitations

* Domain-specific responses
* Limited general knowledge
* No conversation memory

---

# 🚀 Future Improvements

* Add chat memory (multi-turn conversations)
* Expand dataset
* Deploy on Hugging Face Spaces
* Improve UI/UX
* Add multilingual support

---

# 🧩 End-to-End Pipeline

```id="d2z3yx"
Dataset → QLoRA Training → Save Adapter → Hugging Face → Gradio UI
```

---

# 🙌 Conclusion

This project shows how:

* Large models can be fine-tuned efficiently
* Domain-specific AI systems can be built
* Full ML pipelines can be deployed using free resources

---

# 📎 Credits

* Mistral AI (Base Model)
* Hugging Face
* Google Colab
* DeepLearning.AI

---
