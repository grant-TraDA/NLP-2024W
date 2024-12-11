from typing import Mapping

import pandas as pd
from datasets import Dataset

COLUMNS = [
    "statement_id",
    "label",
    "statement",
    "subject",
    "speaker",
    "speaker_job",
    "state",
    "party",
    "true_cnt",
    "false_cnt",
    "half_true_cnt",
    "mostly_true_cnt",
    "pants_on_fire_cnt",
    "context",
]
LABEL_MAPPING = {
    "false": 1,
    "pants-fire": 1,
    "mostly-true": 0,
    "true": 0,
}
STATEMENT_COLUMN = "statement"
LABEL_COLUMN = "label"


def read_data(
    path: str,
    columns: list[str] = COLUMNS,
    label_mapping: Mapping[str, int] = LABEL_MAPPING,
) -> pd.DataFrame:
    df = pd.read_table(path, header=None, names=columns)
    df = df[[STATEMENT_COLUMN, LABEL_COLUMN]]
    df[LABEL_COLUMN] = df[LABEL_COLUMN].map(label_mapping)
    df = df.loc[~df[LABEL_COLUMN].isnull()]
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)
    assert (df.isna().sum() == 0).all()
    assert (
        df[LABEL_COLUMN].drop_duplicates().sort_values().values == [0, 1]
    ).all()
    df = df.rename(columns={STATEMENT_COLUMN: "text"})
    return df


def prepare_data_for_fine_tuning(df: pd.DataFrame, tokenizer):
    dataset = Dataset.from_pandas(df)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            max_length=256,
            truncation=True,
        )

    dataset = dataset.map(tokenize)
    dataset.set_format(
        "torch", columns=["input_ids", "attention_mask", "label"]
    )
    return dataset
