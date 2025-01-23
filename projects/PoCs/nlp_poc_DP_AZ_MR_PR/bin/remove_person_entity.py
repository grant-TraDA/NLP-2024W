from functools import partial
from pathlib import Path

import pandas as pd
import spacy
from loguru import logger

PLACEHOLDER = "__PERSON__"

DATA_PATHS = [
    "data/split_raw/train.tsv",
    "data/split_raw/test.tsv",
    "data/split_raw/valid.tsv",
]

TEXT_COL = 2


def remove_ner(text: str, nlp) -> str:
    new_text = text

    doc = nlp(text)
    names_ent = filter(lambda ent: ent.label_ == "PERSON", doc.ents)
    names = map(lambda ent: ent.text, names_ent)
    for name in names:
        new_text = new_text.replace(name, PLACEHOLDER)

    return new_text


def main() -> None:

    nlp = spacy.load("en_core_web_lg")
    remove_ner_ = partial(remove_ner, nlp=nlp)

    for data_path in DATA_PATHS:
        logger.info(f"start data: {data_path}")
        path = Path(data_path)
        df = pd.read_csv(path, sep="\t")
        texts = df.iloc[:, TEXT_COL]
        new_texts = texts.apply(remove_ner_)
        df.iloc[:, TEXT_COL] = new_texts
        new_data_path = path.parent / f"{path.stem}_removed_ner.tsv"
        df.to_csv(new_data_path, index=False, sep="\t")


if __name__ == "__main__":
    main()
