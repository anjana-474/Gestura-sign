import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments

MODEL_NAME = "facebook/bart-base"

# Load data
df = pd.read_csv("conversation_data.csv")
dataset = Dataset.from_pandas(df)

# Load model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

def preprocess(batch):
    inputs = tokenizer(batch["input"], padding="max_length", truncation=True, max_length=32)
    outputs = tokenizer(batch["output"], padding="max_length", truncation=True, max_length=32)
    inputs["labels"] = outputs["input_ids"]
    return inputs

dataset = dataset.map(preprocess, batched=True)
dataset = dataset.train_test_split(test_size=0.2)


args = TrainingArguments(
    output_dir="./conv_model",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=25,
    logging_steps=10,
    save_total_limit=1,
    report_to="none"
)


trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"]
)

trainer.train()

model.save_pretrained("conversation_model")
tokenizer.save_pretrained("conversation_model")
