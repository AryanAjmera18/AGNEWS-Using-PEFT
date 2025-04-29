# backend/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from train import fine_tune_lora

app = FastAPI()

class TrainRequest(BaseModel):
    lora_rank: int
    lora_alpha: int
    batch_size: int
    epochs: int
    max_length: int

@app.post("/train")
async def train_model(config: TrainRequest):
    config_dict = config.dict()
    run_id = fine_tune_lora(config_dict)
    return {"message": "Training started and completed", "mlflow_run_id": run_id}
