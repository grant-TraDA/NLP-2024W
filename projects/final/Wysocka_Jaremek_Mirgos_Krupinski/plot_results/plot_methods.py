import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_models_results(models, models_names):
    models_results = {}
    
    for model, name in zip(models, models_names):
        df = pd.DataFrame(model, columns=["Lektura", "BLEU", "METEOR", "ROUGE_1", "BERTScore (F1)"])
        scores = df["BERTScore (F1)"].tolist()
        models_results[name] = scores

    return models_results