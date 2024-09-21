import torch
import json
from transformers import AutoTokenizer
from utils import _load_dataset
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

def tokenize(tokenizer, text, max_length):
    res = tokenizer(text, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
    res = {k:v.to("cuda:0") for k, v in res.items()}
    return res

def _get_case_concepts(case):
    legal_concepts = [case["legal_concept"]["criminal_circumstance"][0][0]]
    for item in case["legal_concept"]["sentence_circumstance"]:
        legal_concepts.append(item[0])
    return legal_concepts


def load_model():
    print("load tokenzier function of legal concept predictor...")
    tokenizer = AutoTokenizer.from_pretrained(f"modelfiles/lawformer")
    print("load legal concept predictor...")
    predictor = torch.load(f"ckp/concept_pred_ckp_16000.pkl")
    print("load legal concept label mapping...")
    with open(f"datasets/lc3vg/legal_concept_label_map.json", "r", encoding="utf-8") as fi:
        label_map = json.loads(fi.read())
    return tokenizer, predictor, label_map

def eval_concept_predictor():
    tokenizer, predictor, label_map = load_model()
    dataset = _load_dataset("datasets/lc3vg/llm_cases.json")
    y_pred = []
    y_true = []
    for case in dataset:
        fact = case['bg_info']+case["fact"]
        labels = [0]*len(label_map)
        for c in  _get_case_concepts(case):
            labels[label_map.index(c)] = 1
        y_true.append(labels)
        tokenized_fact = tokenize(tokenizer=tokenizer, text=fact, max_length=1000)
        logits = predictor(**tokenized_fact)["logits"]
        preds = logits>-0.5
        temp = []
        for p, y in zip(preds.tolist()[0], label_map):
            if p:
                temp.append(y)
        labels = [0]*len(label_map)
        for i in temp:
            labels[label_map.index(i)] = 1 
        y_pred.append(labels)
    acc = f"Acc: {round(accuracy_score(y_pred=y_pred, y_true=y_true), 4)} | "
    p = f"P: {round(precision_score(y_pred=y_pred, y_true=y_true, average='samples'), 4)} | "
    r = f"R: {round(recall_score(y_pred=y_pred, y_true=y_true, average='samples'), 4)} | "
    f1 = f"F1: {round(f1_score(y_pred=y_pred, y_true=y_true, average='samples'), 4)}"
    return print(acc + p + r + f1)

if __name__=="__main__":
    eval_concept_predictor()