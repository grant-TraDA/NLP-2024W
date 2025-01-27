# NLP project - bias detection

## Full reproduction

```{bash}
conda create -n nlp2024 python=3.10
conda activate nlp2024
pip install -r requirements.txt
python -m spacy download en_core_web_lg
./train_models.sh
notebooks/results.ipynb # manually
notebooks/results_analysis.ipynb # manually
```

## Model training notes

* Install dependencies - `pip install -r requirements.txt`
* Download required spacy model - `python -m spacy download en_core_web_lg`
* Training all models - `./train_models.sh`
* Training output should be in `output` directory

## Data notes

Preprocessed data is already uploaded to this repository due to its small size. The preprocessing is defined in these files:

* `engine/data.py`
* `notebooks/isot.ipynb`
* `notebooks/coaid.ipynb`

## Results notes

* Finish model training - this should result in a folder output with models partitioned by dataset, model, training type and run number like:

```{bash}
output
    | - coaid
        | - ernie
            | - masked
                | - 1
                    | - test_acc.json
                    | - model_final
                        | - model.safetensors
```

* Install dependencies - if model training was done elsewhere
* run notebooks/results.ipynb - to obtain raw results in json
* change the utilised json results in results_analysis.ipynb
* run notebooks/results_analysis.ipynb - to obtain the figures
