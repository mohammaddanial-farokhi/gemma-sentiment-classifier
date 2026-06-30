<!-- ========================================================= -->
<!--                     HEADER                                -->
<!-- ========================================================= -->

<div align="center">

# 🎬 Gemma Sentiment Classifier

### Fine-Tuning Google's Gemma-2B-IT using QLoRA (4-bit Quantization + LoRA) for IMDB Sentiment Analysis

<br>

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red.svg)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![PEFT](https://img.shields.io/badge/PEFT-LoRA-green)
![BitsAndBytes](https://img.shields.io/badge/Quantization-4Bit-orange)
![Dataset](https://img.shields.io/badge/Dataset-IMDB-purple)
![Task](https://img.shields.io/badge/Task-Sentiment%20Analysis-success)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

</div>

---


# 🚀 Overview

This repository demonstrates an efficient approach to fine-tuning **Google's Gemma-2B-IT** for **binary sentiment classification** using the IMDB movie review dataset.

Instead of updating all **2 billion parameters**, the project leverages **Parameter-Efficient Fine-Tuning (PEFT)** through **LoRA adapters**, combined with **4-bit quantization (QLoRA)**, dramatically reducing GPU memory requirements while preserving strong predictive performance.

The implementation follows a fully modular pipeline that covers:

- Model loading
- Dataset preparation
- Tokenization
- Training
- Evaluation
- Adapter saving
- Adapter loading
- Inference

---

# 💡 Motivation

Large Language Models have shown outstanding performance across many NLP tasks.

However, full fine-tuning requires:

- Large GPUs
- Huge memory consumption
- Long training times

This project explores a much more efficient alternative:

> Fine-tuning only a tiny subset of parameters while keeping the original Gemma model frozen.

The result is a lightweight sentiment classifier that reaches approximately **90% accuracy** while using only **10% of the IMDB training dataset**.

---

## ✨ Features

| 🤖 Model | ⚡ Training | 📈 Evaluation | 🚀 Inference |
|:---------|:-----------|:--------------|:-------------|
| Gemma-2B-IT | LoRA (PEFT) | Accuracy | Single Prediction |
| 4-bit QLoRA | Early Stopping | Precision | Batch Prediction |
| Sequence Classification | Dynamic Padding | Recall | Adapter Reloading |
| HuggingFace Transformers | 10% IMDB Dataset | F1 Score | Modular Pipeline |
---

# 🏗 Project Architecture

```

                    +----------------------+
                    |      IMDB Dataset    |
                    +----------+-----------+
                               |
                               |
                               ▼
                 Dataset Sampling (10%)
                               |
                               ▼
                    Text Tokenization
                               |
                               ▼
                  Dynamic Padding
                               |
                               ▼
             Gemma-2B-IT (4-bit Quantized)
                               |
                               ▼
                LoRA Adapter Training
                               |
                               ▼
                  Fine-tuned Adapter
                               |
                               ▼
                 Save Adapter (.PEFT)
                               |
                               ▼
               Load Adapter for Inference
                               |
                               ▼
               Sentiment Prediction

```

---





# 🧠 Code Organization

The entire implementation is written in a modular manner.

| Module | Responsibility |
|----------|----------------|
| `hug_login()` | Authenticate with HuggingFace |
| `load_model_and_tokenizer()` | Load Gemma-2B model using 4-bit quantization |
| `load_imdb_dataset()` | Download IMDB dataset |
| `create_small_dataset()` | Sample only 10% of dataset |
| `tokenize_dataset()` | Tokenize text reviews |
| `get_data_collator()` | Dynamic padding |
| `compute_metrics()` | Accuracy, Precision, Recall, F1 |
| `lora_tune()` | Fine-tune using LoRA |
| `load_lora_model()` | Reload trained adapter |
| `predict_single()` | Predict custom review |
| `evaluate_on_sample()` | Evaluate random reviews |
| `print_single_prediction()` | Pretty prediction output |
| `print_evaluation_summary()` | Evaluation summary |

---

# 🛠 Technologies

| Category | Library |
|------------|----------------|
| Language | Python |
| Deep Learning | PyTorch |
| LLM | Transformers |
| PEFT | PEFT |
| Quantization | BitsAndBytes |
| Dataset | HuggingFace Datasets |
| Metrics | Evaluate |
| Numerical | NumPy |
| DataFrame | Pandas |
| Authentication | HuggingFace Hub |

---

# 📊 Dataset

Dataset:

> IMDB Movie Reviews : https://huggingface.co/datasets/stanfordnlp/imdb

Binary Classification: Positive - Negative

The original dataset contains **50,000 movie reviews**.

To reduce computational cost, only **10%** of the training split is used.

---

# 🤖 Base Model

Google

Gemma-2B-IT

Task

Sequence Classification

Output Classes

- POSITIVE
- NEGATIVE

The base model is loaded directly from HuggingFace and converted into a sequence classification model with two output labels.

---

# 🔥 QLoRA

Instead of fine-tuning all model parameters, this project adopts **QLoRA**, which combines:

- 4-bit NormalFloat Quantization (NF4)

- Double Quantization

- LoRA Adapters

Benefits:

✔ Lower VRAM

✔ Faster Training

✔ Lower Storage

✔ High Accuracy

✔ Consumer GPU Friendly

---


# 🧠 LoRA Configuration

Only the attention projection matrices are trained.

| Parameter | Value |
|------------|-------|
| Rank | 64 |
| Alpha | 32 |
| Dropout | 0.05 |
| Bias | None |
| Task Type | Sequence Classification |

Target Modules:

```
q_proj
k_proj
v_proj
o_proj
```

This significantly reduces the number of trainable parameters compared to full fine-tuning.

---
# ⚙️ Training Configuration

The model is fine-tuned using the Hugging Face `Trainer` API with **Parameter-Efficient Fine-Tuning (PEFT)** and **QLoRA**.

## Hyperparameters

| Parameter | Value |
|------------|-------|
| Base Model | Google Gemma-2B-IT |
| Fine-Tuning Method | LoRA |
| Quantization | 4-bit NF4 |
| Optimizer | AdamW (Trainer Default) |
| Learning Rate | 2e-5 |
| Epochs | 5 |
| Train Batch Size | 4 |
| Evaluation Batch Size | 1 |
| Weight Decay | 0.01 |
| Precision | FP16 |
| Early Stopping | Enabled |
| Save Strategy | Every Epoch |
| Evaluation Strategy | Every Epoch |
| Best Model Selection | Validation Loss |

---

# 🧩 Quantization Configuration

The project uses **BitsAndBytes** to reduce GPU memory usage.

```python
load_in_4bit = True
bnb_4bit_quant_type = "nf4"
bnb_4bit_use_double_quant = True
bnb_4bit_compute_dtype = torch.bfloat16
```

Benefits include:

- Reduced GPU memory footprint
- Faster model loading
- Faster fine-tuning
- Ability to train on consumer GPUs



---

# 📈 Evaluation Metrics

Model performance is evaluated using four commonly used classification metrics.

| Metric | Description |
|---------|-------------|
| Accuracy | Overall correctness |
| Precision | Correct positive predictions |
| Recall | Coverage of actual positives |
| F1 Score | Balance between Precision and Recall |

These metrics are automatically computed after every evaluation epoch.

---

# 💻 Installation

Clone the repository

```bash
git clone https://github.com/yourusername/gemma-sentiment-classifier.git

cd gemma-sentiment-classifier
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate environment

Windows

```bash
.venv\Scripts\activate
```

Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🔑 HuggingFace Authentication

Create a `.env` file in the project root.

```
HF_TOKEN=your_huggingface_access_token
```

The project automatically authenticates using

```
python-dotenv
```

---

# 💬 Example

Input

```text
The movie was one of the best films I've ever watched.
```

Output

```text
Predicted Label : POSITIVE

Probability Positive : 0.996

Probability Negative : 0.004
```

---

# 📊 Sample Evaluation Output

```
================================================

EVALUATION ON 10 SAMPLES

================================================

Accuracy : 0.90

================================================
```

---

# 💾 Saved Artifacts

After training, the following files are generated.

```
peft-gemma-imdb/

adapter_config.json

adapter_model.safetensors

tokenizer.json

tokenizer_config.json

special_tokens_map.json
```

Only the LoRA adapter is saved.

The original Gemma model remains unchanged.

---

# 📚 What I Learned

Through this project I explored:

- Large Language Model fine-tuning
- Hugging Face ecosystem
- PEFT
- QLoRA
- BitsAndBytes Quantization
- Dataset preprocessing
- Evaluation metrics
- Modular ML project architecture
- Inference pipeline design

---

# 🔮 Future Improvements

- Multi-class sentiment classification
- Hyperparameter optimization
- Full IMDB training
- Hugging Face Hub deployment
- Gradio Web Interface
- Streamlit Dashboard
- Docker support
- Model benchmarking
- ONNX export
- TensorRT optimization

---

# 📖 References

Google Gemma

PEFT

Transformers

BitsAndBytes

HuggingFace Datasets

---

# 🤝 Contributing

Contributions are welcome.

Feel free to open an Issue or submit a Pull Request.

---

# 📄 License

This project is licensed under the MIT License.

---

# 👨‍💻 Author

**Mohammad Danial Farokhi**

Machine Learning Engineer

Natural Language Processing

Large Language Models

Deep Learning

GitHub:

https://github.com/mohammaddanial-farokhi

If you found this project useful, consider giving it a ⭐.
