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

# 📖 Table of Contents

- Overview
- Motivation
- Features
- Project Architecture
- Pipeline
- Project Structure
- Technologies
- Dataset
- Model
- Quantization
- LoRA Fine-Tuning
- Training Configuration
- Evaluation Metrics
- Installation
- Usage
- Training
- Inference
- Results
- Future Work
- References
- License

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

# ✨ Features

✅ Google Gemma-2B-IT

✅ QLoRA (4-bit Quantization)

✅ PEFT LoRA Adapters

✅ Automatic HuggingFace Authentication

✅ Modular Codebase

✅ Dynamic Dataset Sampling

✅ Tokenization Pipeline

✅ Dynamic Padding

✅ Early Stopping

✅ Automatic Evaluation

✅ Accuracy

✅ Precision

✅ Recall

✅ F1 Score

✅ Automatic Adapter Loading

✅ Single Text Prediction

✅ Batch Evaluation

✅ Easily Extendable

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

# ⚙ Project Pipeline

```

Load HF Token
      │
      ▼

Load Gemma-2B
      │
      ▼

Load IMDB Dataset
      │
      ▼

Sample 10% Dataset
      │
      ▼

Tokenization
      │
      ▼

Data Collator
      │
      ▼

Evaluation Metrics
      │
      ▼

Prepare Model for QLoRA
      │
      ▼

LoRA Configuration
      │
      ▼

Trainer
      │
      ▼

Training
      │
      ▼

Save Adapter
      │
      ▼

Load Adapter
      │
      ▼

Predict

```

---

# 📂 Project Structure

```
gemma-sentiment-classifier/

│

├── train.py

├── peft-gemma-imdb/

│      ├── adapter_model.safetensors
│      ├── adapter_config.json
│      ├── tokenizer.json
│      └── ...

├── requirements.txt

├── .env

└── README.md

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

> IMDB Movie Reviews

Binary Classification

Positive

Negative

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

# 🎯 LoRA Configuration

| Parameter | Value |
|------------|--------|
| Rank (r) | 64 |
| Alpha | 32 |
| Dropout | 0.05 |
| Bias | none |
| Task | Sequence Classification |
| Target Modules | q_proj, k_proj, v_proj, o_proj |
