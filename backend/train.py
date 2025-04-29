# backend/train.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
import evaluate
import mlflow
import numpy as np
from torch.utils.data import DataLoader

def fine_tune_lora(config: dict):
    # Start MLflow run
    mlflow.set_experiment("AGNews-LoRA")
    run = mlflow.start_run()
    
    mlflow.log_params(config)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load dataset
    dataset = load_dataset("ag_news")
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    
    def preprocess(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=config["max_length"])
    
    encoded_dataset = dataset.map(preprocess, batched=True)
    encoded_dataset = encoded_dataset.rename_column("label", "labels")
    encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    train_dataloader = DataLoader(encoded_dataset["train"], batch_size=config["batch_size"], shuffle=True)
    eval_dataloader = DataLoader(encoded_dataset["test"], batch_size=config["batch_size"])

    # Load base model and apply LoRA
    base_model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=4)
    lora_config = LoraConfig(
        r=config["lora_rank"],
        lora_alpha=config["lora_alpha"],
        target_modules=["query", "value"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_CLS
    )
    model = get_peft_model(base_model, lora_config)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    # Training Loop
    for epoch in range(config["epochs"]):
        model.train()
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Evaluation after each epoch
        model.eval()
        preds, labels = [], []
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

    run_id = run.info.run_id
    mlflow.end_run()
    return run_id
