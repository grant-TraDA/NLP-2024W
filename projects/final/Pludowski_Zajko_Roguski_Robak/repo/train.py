import json
import shutil
from pathlib import Path

import pandas as pd
from loguru import logger
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from engine.args import get_arguments
from engine.data import prepare_data_for_fine_tuning
from engine.metrics import compute_metrics
from engine.replace_persons import replace_ner

tqdm.pandas()

MODEL_MAPPING = {
    "roberta": "roberta-base",
    "ernie": "nghuyong/ernie-2.0-base-en",
}


def main():
    args = get_arguments()
    logger.info(f"Parsed arguments: {args.__dict__}")

    logger.info("Preparing data data")
    data_path = Path("data") / args.data
    train_df = pd.read_csv(data_path / "train.csv")
    valid_df = pd.read_csv(data_path / "valid.csv")
    test_df = pd.read_csv(data_path / "test.csv")

    if args.mask == "yes":
        train_df["text_masked"] = train_df["text"].progress_apply(replace_ner)
        valid_df["text_masked"] = valid_df["text"].progress_apply(replace_ner)
        test_df["text_masked"] = test_df["text"].progress_apply(replace_ner)

    logger.info("Loading model")
    model_name = MODEL_MAPPING[args.model]
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, config=config
    )

    logger.info("Starting fine-tuning")
    train_dataset = prepare_data_for_fine_tuning(train_df, tokenizer)
    valid_dataset = prepare_data_for_fine_tuning(valid_df, tokenizer)
    test_dataset = prepare_data_for_fine_tuning(test_df, tokenizer)

    output_dir = (
        Path("output")
        / args.data
        / args.model
        / ("masked" if args.mask == "yes" else "unmasked")
        / str(args.seed)
    )
    assert not output_dir.exists(), f"{output_dir} already exists"
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=10,
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
        seed=args.seed,
        greater_is_better=True,
        metric_for_best_model="eval_accuracy",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=2, early_stopping_threshold=0.01
            )
        ],
    )

    trainer.train()

    logger.info("Saving trained model")
    model.save_pretrained(output_dir / "model_final")

    logger.info("Evaluating on test data")
    test_acc = trainer.evaluate(test_dataset)["eval_accuracy"]
    with open(output_dir / "test_acc.json", "w") as f:
        json.dump(test_acc, f)

    logger.info("Deleting checkpoints")
    list(
        map(
            lambda p: shutil.rmtree(p),
            output_dir.rglob("checkpoint*"),
        )
    )


if __name__ == "__main__":
    main()
