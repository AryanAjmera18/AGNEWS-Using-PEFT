import streamlit as st
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
import evaluate
import mlflow
import numpy as np
import streamlit as st
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

import streamlit as st
# (then continue normal imports)
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
# etc.

# Always display sidebar and title immediately
st.set_page_config(page_title="Fine-tune RoBERTa with LoRA", page_icon="🔵")
st.title("🔵 RoBERTa Fine-Tuning with LoRA - AGNews")

st.sidebar.header("LoRA Configuration")
lora_rank = st.sidebar.slider("LoRA Rank (r)", 2, 16, 8)
lora_alpha = st.sidebar.slider("LoRA Alpha (scaling)", 8, 64, 16)
epochs = st.sidebar.slider("Epochs", 1, 5, 3)
batch_size = st.sidebar.selectbox("Batch Size", [4, 8, 16], index=1)
max_length = st.sidebar.slider("Max Sequence Length", 64, 256, 128)

# THEN below, inside the button
if st.button("🚀 Start Fine-Tuning"):
    # training code here
    pass


# Page Title
st.title("🔵 RoBERTa Fine-Tuning with LoRA - AGNews")

# Sidebar - Model configs
st.sidebar.header("LoRA Configuration")
lora_rank = st.sidebar.slider("LoRA Rank (r)", 2, 16, 8)
lora_alpha = st.sidebar.slider("LoRA Alpha (scaling)", 8, 64, 16)
epochs = st.sidebar.slider("Epochs", 1, 5, 3)
batch_size = st.sidebar.selectbox("Batch Size", [4, 8, 16], index=1)
max_length = st.sidebar.slider("Max Sequence Length", 64, 256, 128)

# Button to start
if st.button("🚀 Start Fine-Tuning"):
    st.write("Preparing...")

    # MLflow setup
    mlflow.set_experiment("RoBERTa-LoRA-AGNews-Streamlit")
    mlflow.start_run()
    
    mlflow.log_params({
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "epochs": epochs,
        "batch_size": batch_size,
        "max_length": max_length
    })

    # Load dataset
    dataset = load_dataset("ag_news")
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    def preprocess(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=max_length)

    encoded_dataset = dataset.map(preprocess, batched=True)
    encoded_dataset = encoded_dataset.rename_column("label", "labels")
    encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # Load base model
    base_model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=4)

    # Apply LoRA
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["query", "value"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_CLS
    )
    model = get_peft_model(base_model, lora_config)

    # Metrics
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
        f1 = f1_metric.compute(predictions=preds, references=labels, average="weighted")["f1"]
        mlflow.log_metrics({"eval_accuracy": acc, "eval_f1": f1})
        return {"accuracy": acc, "f1": f1}

    # Prepare training manually
    train_dataloader = DataLoader(encoded_dataset["train"], batch_size=batch_size, shuffle=True)
    eval_dataloader = DataLoader(encoded_dataset["test"], batch_size=batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    progress_bar = st.progress(0)
    chart_data = {"accuracy": [], "f1": [], "epoch": []}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    total_steps = epochs * len(train_dataloader)
    step_count = 0

    st.write("Training started...")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item()
            step_count += 1
            progress_bar.progress(step_count / total_steps)

        # After epoch - evaluate
        model.eval()
        preds = []
        labels = []
        with torch.no_grad():
            for batch in eval_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                logits = outputs.logits
                preds.append(logits.argmax(dim=-1).cpu())
                labels.append(batch["labels"].cpu())
        
        preds = torch.cat(preds)
        labels = torch.cat(labels)
        
        acc = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
        f1 = f1_metric.compute(predictions=preds, references=labels, average="weighted")["f1"]

        mlflow.log_metrics({f"epoch_{epoch+1}_accuracy": acc, f"epoch_{epoch+1}_f1": f1})
        
        st.metric(label=f"Epoch {epoch+1} Accuracy", value=f"{acc:.4f}")
        st.metric(label=f"Epoch {epoch+1} F1 Score", value=f"{f1:.4f}")

        chart_data["accuracy"].append(acc)
        chart_data["f1"].append(f1)
        chart_data["epoch"].append(epoch+1)

    st.success("🎯 Fine-Tuning Completed!")
    run_id = mlflow.active_run().info.run_id
    mlflow.end_run()

    st.success(f"✅ MLflow Run ID: {run_id}")

    # Plot Accuracy/F1 vs Epoch
    import pandas as pd
    import altair as alt

    chart_df = pd.DataFrame(chart_data)
    chart_df = chart_df.melt(id_vars="epoch", value_vars=["accuracy", "f1"], var_name="Metric", value_name="Value")

    st.altair_chart(
        alt.Chart(chart_df)
        .mark_line(point=True)
        .encode(
            x="epoch:O",
            y="Value:Q",
            color="Metric:N",
            tooltip=["epoch", "Metric", "Value"]
        )
        .properties(title="Training Metrics over Epochs")
        .interactive()
    )
    st.balloons()
