import huggingface_hub
from huggingface_hub import login
from dotenv import load_dotenv
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    BitsAndBytesConfig,
    TrainingArguments,
    EarlyStoppingCallback,
    Trainer,
)
from peft import (
    LoraConfig,
    PeftModel,
    prepare_model_for_kbit_training,
    get_peft_model,
)
import os
import random
import evaluate
import numpy as np
import torch
import pandas as pd


###----------------------##
## 1-loding base model
###----------------------##
def hug_login():
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    login(token=hf_token)
    # user_info = huggingface_hub.whoami()
    # print(user_info)


def load_model_and_tokenizer(model_id="google/gemma-2b-it"):
    hug_login()
    os.environ["HF_HOME"] = "D:/hf_cache"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path="google/gemma-2b-it",  # "google/gemma-2b-it"
        num_labels=2,  # Number of output labels (2 for binary sentiment classification)
        id2label={0: "NEGATIVE", 1: "POSITIVE"},
        label2id={"NEGATIVE": 0, "POSITIVE": 1},
        quantization_config=bnb_config,
        device_map="auto",
    )

    return model, tokenizer


###----------------------##
## 2-load and preprocess dataset
###----------------------##
def load_imdb_dataset(name="stanfordnlp/imdb"):
    return load_dataset(name)


def create_small_dataset(dataset, rate=0.1, seed=42):
    random.seed(seed)
    num_train = int(rate * len(dataset["train"]))
    num_test = int(rate * len(dataset["test"]))

    small_train = dataset["train"].shuffle(seed=seed).select(range(num_train))
    small_test = dataset["test"].shuffle(seed=seed).select(range(num_test))

    return DatasetDict({"train": small_train, "test": small_test})


def tokenize_dataset(dataset, tokenizer, max_length=512):
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_length)

    return dataset.map(preprocess_function, batched=True)


###----------------------##
## 4-data collator Preparation
###----------------------##
def get_data_collator(tokenizer):
    return DataCollatorWithPadding(tokenizer=tokenizer)


###----------------------##
## 5-Evaluation Metrics
###----------------------##
metric = evaluate.combine(["accuracy", "f1", "precision", "recall"])


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)  # Convert probabilities to predicted labels
    return metric.compute(predictions=predictions, references=labels)


###----------------------##
## 6-Fine-Tuning with LoRA Adapter
###----------------------##


def lora_tune(model, tokenizer, tokenized_imdb, data_collator, compute_metrics):
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=64,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir="epoch_weights",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=1,
        num_train_epochs=5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=2,
        fp16=True,
        push_to_hub=False,
        report_to="none",
        metric_for_best_model="eval_loss",
    )
    early_stop = EarlyStoppingCallback(early_stopping_patience=1, early_stopping_threshold=0.0)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_imdb["train"],
        eval_dataset=tokenized_imdb["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[early_stop],
    )

    trainer.train()

    peft_model_path = "./peft-gemma-imdb"

    trainer.model.save_pretrained(peft_model_path)
    tokenizer.save_pretrained(peft_model_path)


###----------------------##
## 7-Loading LoRA
###----------------------##
def load_lora_model(base_model_id="google/gemma-2b-it", lora_path="./peft-gemma-imdb"):

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    base_model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=base_model_id,
        num_labels=2,
        id2label={0: "NEGATIVE", 1: "POSITIVE"},
        label2id={"NEGATIVE": 0, "POSITIVE": 1},
        quantization_config=bnb_config,
        device_map="auto",
    )

    lora_model = PeftModel.from_pretrained(base_model, lora_path)

    return lora_model, tokenizer


###----------------------##
## 8-predict
###----------------------##
def predict_single(text, model, tokenizer, device="cuda"):

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=-1)
    pred_label = torch.argmax(probs, dim=-1).item()
    prob_positive = probs[0, 1].item()
    prob_negative = probs[0, 0].item()
    label_name = "POSITIVE" if pred_label == 1 else "NEGATIVE"

    return {
        "text": text,                     
        "pred_label": pred_label,
        "label_name": label_name,         
        "prob_negative": prob_negative,
        "prob_positive": prob_positive,
    }


def evaluate_on_sample(model, tokenizer, num_samples=10):
    dataset = load_dataset("imdb")
    df_test = pd.DataFrame(dataset["test"]).head(num_samples)

    predictions = df_test["text"].apply(lambda x: predict_single(x, model, tokenizer))

    df_test["y_pred"] = predictions.apply(lambda x: x["pred_label"])
    df_test["prob_positive"] = predictions.apply(lambda x: x["prob_positive"])

    accuracy = (df_test["y_pred"] == df_test["label"]).mean()
    print(f"Model Accuracy on {num_samples} samples: {accuracy:.4f}")

    return df_test


###----------------------##
## 9-print results
###----------------------##


def print_single_prediction(result):
    print("\n" + "=" * 60)
    print("SINGLE PREDICTION RESULT")
    print("=" * 60)
    print(f"Text: {result['text']}")
    print(f"Predicted Label: {result['label_name']} (Class {result['pred_label']})")
    print(f"Probability - Negative: {result['prob_negative']:.4f}")
    print(f"Probability - Positive: {result['prob_positive']:.4f}")
    print("=" * 60)


def print_evaluation_summary(df_result, num_samples):
    accuracy = (df_result["y_pred"] == df_result["label"]).mean()

    print("\n" + "=" * 60)
    print(f"EVALUATION ON {num_samples} SAMPLES")
    print("=" * 60)
    print(f"Model Accuracy: {accuracy:.4f}")
    print("\nSample Results (First 5):")
    print(df_result[["text", "label", "y_pred", "prob_positive"]].head().to_string(index=False))
    print("=" * 60)


if __name__ == "__main__":
    lora_path = "./peft-gemma-imdb"

    if os.path.exists(lora_path):
        print("LoRA model already exists. Loading...")
        lora_model, tokenizer = load_lora_model(lora_path=lora_path)
        print("model loaded")

    else:
        print("LoRA model not found. Starting training...")

        model, tokenizer = load_model_and_tokenizer()

        full_dataset = load_imdb_dataset()
        small_dataset = create_small_dataset(full_dataset, rate=0.1)
        tokenized_dataset = tokenize_dataset(small_dataset, tokenizer)

        data_collator = get_data_collator(tokenizer)

        lora_tune(
            model=model,
            tokenizer=tokenizer,
            tokenized_imdb=tokenized_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        print("Training completed. Model saved.")

        sample_text = "The movie was the best movie I have ever seen!!!"
        single_test = predict_single(sample_text, model, tokenizer)
        print_single_prediction(single_test)

        multi_test = evaluate_on_sample(model, tokenizer)
        print_evaluation_summary(multi_test, 10)
