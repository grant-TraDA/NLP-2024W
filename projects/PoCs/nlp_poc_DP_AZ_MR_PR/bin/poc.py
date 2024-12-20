import datetime

import evaluate
import numpy as np
from loguru import logger
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from engine.data import prepare_data_for_fine_tuning, read_data

MODEL_ID = "roberta-base"

logger.info("Loading data")
train = read_data("data/split_raw/train.tsv")
valid = read_data("data/split_raw/valid.tsv")
test = read_data("data/split_raw/test.tsv")

logger.info("Loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

logger.info("Preparing data for fine-tuning")
train_dataset = prepare_data_for_fine_tuning(train, tokenizer)
valid_dataset = prepare_data_for_fine_tuning(valid, tokenizer)
test_dataset = prepare_data_for_fine_tuning(test, tokenizer)


logger.info("Preparing model for fine-tuning")
metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


config = AutoConfig.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, config=config)
timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
training_args = TrainingArguments(
    output_dir=f"output/{timestamp}",
    num_train_epochs=15,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=128,
    eval_strategy="epoch",
    logging_strategy="steps",
    logging_steps=10,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=500,
    save_strategy="epoch",
    load_best_model_at_end=True,
    save_total_limit=2,
    report_to="tensorboard",
    seed=123,
    greater_is_better=True,
    metric_for_best_model="eval_accuracy",
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    compute_metrics=compute_metrics,
)
logger.info("Training model")
trainer.train()
trainer.save_model(
    f"output/{timestamp}/model",
)
