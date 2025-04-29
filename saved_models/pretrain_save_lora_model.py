# pretrain_save_lora.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
import evaluate
from torch.utils.data import DataLoader

# Configuration
config = {
    "lora_rank": 8,
    "lora_alpha": 16,
    "batch_size": 8,
    "epochs": 3,
    "max_length": 128
}

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

# Load base model
base_model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=4)

# Apply LoRA
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

# Training loop
for epoch in range(config["epochs"]):
    model.train()
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# Save the LoRA adapter only
model.save_pretrained("./saved_models/lora_rank8_alpha16")

print("✅ LoRA Adapter Saved at ./saved_models/lora_rank8_alpha16")
