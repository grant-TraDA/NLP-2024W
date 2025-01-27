from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
    TextClassificationPipeline,
)
import numpy as np
import pandas as pd
import spacy
from pathlib import Path

from .data import prepare_data_for_fine_tuning
from .ner_detector import tokenize_evaluate_and_detect_NERs


def get_training_data(models_path, parse_accuracies=False):
    all_files = [path for path in models_path.rglob("model_final/model.safetensors")]
    
    models = [(path.parts[2], path.parts[1], path.parts[3], path.parts[4]) for path in all_files]
    model_df = pd.DataFrame(models, columns=["model", "dataset", "training_type", "run"])
    if parse_accuracies:
        accuracies = []
        for file in all_files:
            with open(file.parent.with_name("test_acc.json"), "r") as f:
                accuracies.append(float(f.readline().strip()))
        model_df["accuracy"] = accuracies
    
    return model_df


def get_person_relative_importance(pipeline, 
                                  test_dataset):
    res2 = tokenize_evaluate_and_detect_NERs(pipeline, 
                                  test_dataset['text'], 
                                  spacy_model="en_core_web_lg",
                                  return_mappings_for_each_text=True)
    ratios = []
    for sentence in res2:
        avg_per_imp = np.array(list(map(lambda x: abs(x[1]), filter(lambda z: z[2] == 'PERSON', sentence)))).mean()
        avg_imp = np.array(list(map(lambda x: abs(x[1]), sentence))).mean()
        ratios.append(avg_per_imp / avg_imp)
        
    return ratios


def get_map_person_importance(res):
    persons = list(map(lambda x: x[0], filter(lambda x: x[2] == 'PERSON', res)))
    importances = list(map(lambda x: x[1], filter(lambda x: x[2] == 'PERSON', res)))
    
    nlp = spacy.load("en_core_web_lg")
    
    persons_unique = {}
    per = ''

    idx = 0
    imp = 0
    cnt = 0
    while idx != len(persons):
        if per == '':
            per = persons[idx]
            imp = importances[idx]
            cnt = 1
            idx += 1
        elif persons[idx][:2] == '##':
            per = per + persons[idx][2:]
            imp += importances[idx]
            cnt += 1
            idx += 1
        else:
            if per not in persons_unique.keys():
                persons_unique[per] = []
            persons_unique[per].append(imp / cnt)
            cnt = 0
            per = ''
            
    new_persons = {}
    for key in persons_unique:
        
        doc = nlp(key)
        is_ok = False
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                is_ok = True
                break
        if is_ok:
            new_persons[key.lower()] = np.mean(persons_unique[key])
    return new_persons


def get_top_persons(persons_unique, negative=False, n=5):
    importance = list(persons_unique.values())
    persons = list(persons_unique.keys())
    importance = np.array(importance)
    persons = np.array(persons)
    if not negative:
        importance = -importance
    top_persons = persons[np.argsort(importance)[:n]]
    return top_persons.tolist()


def pipeline_out_to_vec(pipeline_out):
    preds = []
    for out in pipeline_out:
        if out[0]['label'] == 'LABEL_1':
            preds.append(out[0]['score'])
        else:
            preds.append(out[1]['score'])
    return preds


def find_random_person_words(sentence, persons):
    found = set(sentence.lower().split()).intersection(persons)
    if len(found) == 0:
        return 'NOT EXIST'
    else:
        return list(found)[np.random.choice(len(found))]


def convert_prediction(pred):
    if pred[0]["label"] == "LABEL_1":
        return pred[0]["score"]
    else:
        return pred[1]["score"]


def process_models(models_df, device="cpu"):
    results = {}
    results_misc = {}

    for _, row in models_df.iterrows():
        
        results_misc['-'.join([row["dataset"], row["model"], row["training_type"]])] = {}
        
        print(row["dataset"])
        model_path = Path("output", row["dataset"], row["model"], row["training_type"], row["run"], "model_final", "model.safetensors")
        model_id = "roberta-base" if row["model"] == "roberta" else "nghuyong/ernie-2.0-base-en"
        config = AutoConfig.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, config=config
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        data = pd.read_csv(Path("data", row["dataset"], "test.csv"), header=0)
        test_dataset = prepare_data_for_fine_tuning(data, tokenizer)
        test_dataset = test_dataset.select(np.arange(min(1000, test_dataset.shape[0])))
        model.eval()
        pipeline = TextClassificationPipeline(
            model=model, tokenizer=tokenizer, top_k=2, device=device
        )

        if (device == "cuda"):
            model.cuda()
        else:
            model.cpu()

        ratios = get_person_relative_importance(pipeline, test_dataset)
        results_misc['-'.join([row["dataset"], row["model"], row["training_type"]])]['ratios'] = ratios

        res = tokenize_evaluate_and_detect_NERs(pipeline, 
                                    test_dataset['text'], 
                                    spacy_model="en_core_web_lg")

        person_importance_mapping = get_map_person_importance(res)
        top_positive_persons = get_top_persons(person_importance_mapping, negative=False, n=10)
        top_negative_persons = get_top_persons(person_importance_mapping, negative=True, n=10)

        orig_pred = pipeline_out_to_vec(pipeline(test_dataset["text"]))
        preds = orig_pred

        replacements = []
        test_counterfactuals = test_dataset.to_pandas().copy()

        for ix in test_counterfactuals.index:
            top_per_idx = np.random.choice(10)
            if preds[ix] > 0.5:
                person_to_add = top_negative_persons[top_per_idx]
            else:
                person_to_add = top_positive_persons[top_per_idx]
                
            text = test_counterfactuals.loc[ix, ["text"]].get(0)
            person_to_remove = find_random_person_words(text.lower(), person_importance_mapping.keys())
            test_counterfactuals.loc[ix, ["text"]] = text.lower().replace(person_to_remove, person_to_add)
            
            replacements.append((person_to_remove, person_to_add))

        adv_pred = pipeline_out_to_vec(pipeline(test_counterfactuals["text"].to_list()))

        results['-'.join([row["dataset"], row["model"], row["training_type"]])] = {"orig_pred": orig_pred, "adv_pred": adv_pred, "replacements": replacements}
    return results, results_misc
