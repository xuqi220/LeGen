from rouge_chinese import Rouge
rouge = Rouge()
import random
import evaluate
from evaluate import load
import bert_score
from utils import seg, _load_dataset
import json,re
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


def _rouge_score(gens, views):
    """
    cite: ROUGE: A Package for Automatic Evaluation of Summaries
    gens: list of str
    views: list of str
    """
    assert len(gens) == len(views)
    hyps, refs = [], []
    for g, v in zip(gens, views):
        hyps.append(" ".join(seg.cut(g.replace("\n", ""))))
        refs.append(" ".join(seg.cut(v.replace("\n", ""))))
    scores = rouge.get_scores(hyps, refs, avg=True)
    return scores

def _bleu_score(gens, views):
    """
    cite: BLEU: a Method for Automatic Evaluation of Machine Translation
    gens: list of str
    views: list of str
    """
    hyps, refs = [], []
    for g, v in zip(gens, views):
        hyps.append(" ".join(seg.cut(g.replace("\n", ""))))
        refs.append([" ".join(seg.cut(v.replace("\n", "")))])
    bleu = evaluate.load("bleu")
    results = bleu.compute(predictions=hyps, references=refs)
    scores = {}
    for i, score in enumerate(results["precisions"]):
        scores[f"bleu-{i+1}"] = round(score,4)
    return scores

def _bert_score(gens, views):
    """
    cite: BERTScore: Evaluating Text Generation with BERT
    gens: list of str
    views: list of str
    """
    assert len(gens) == len(views)
    gens = [g.replace("\n", "") for g in gens]
    views = [v.replace("\n", "") for v in views]
    P, R, F1 = bert_score.score(gens, views, lang="zh", verbose=True)
    return round(P.mean().item(),4), round(R.mean().item(),4), round(F1.mean().item(),4)
    
def _legal_concept_score_v1(gens, labeled_legal_concepts):
    """
    Legal Concept Matching Score (Precision)
    gens:list of str
    labeled_legal_concepts:list of list of str
    """
    scores = []
    with open("datasets/legal_concept_label_map.json", "r", encoding="utf-8") as fi:
        all_legal_concepts = json.loads(fi.read())
    for gen, llc in zip(gens, labeled_legal_concepts):
        pred_concepts = [i for i in all_legal_concepts if i in gen]
        # 预测且正确
        a = len(set(llc).intersection(set(pred_concepts)))+0.001
        b = max(len(pred_concepts), len(llc))+0.001
        scores.append(a/b)
    return round(sum(scores)/len(scores), 4)

def _legal_concept_score_v2(gens, labeled_legal_concepts):
    """
    Legal Concept Matching Score (Precision)
    gens:list of str
    labeled_legal_concepts:list of list of str
    """
    with open("datasets/lc3vg/legal_concept_label_map.json", "r", encoding="utf-8") as fi:
        all_legal_concepts = json.loads(fi.read())
    y_trues = []
    y_preds = []
    for gen, llc in zip(gens, labeled_legal_concepts):
        pred_concepts = [1 if i in gen else 0 for i in all_legal_concepts]
        true_concepts = [1 if i in llc else 0 for i in all_legal_concepts]
        y_preds.append(pred_concepts)
        y_trues.append(true_concepts)
    acc = f"Acc: {round(accuracy_score(y_pred=y_preds, y_true=y_trues), 4)} |"
    p = f"P: {round(precision_score(y_pred=y_preds, y_true=y_trues, average='samples'), 4)} |"
    r = f"R: {round(recall_score(y_pred=y_preds, y_true=y_trues, average='samples'), 4)} |"
    f1 = f"F1: {round(f1_score(y_pred=y_preds, y_true=y_trues, average='samples'), 4)}"
    return acc + p + r + f1


def get_gold_views(case):
    gold_views = []
    gold_views.append(case["legal_concept"]["criminal_circumstance"][0][1])
    for item in case["legal_concept"]["sentence_circumstance"]:
        gold_views.append(item[1])
    return gold_views
    
def eval_solver_of_prompt_based_method(data_path, use_verifier=True):
    gens, views, concepts = [], [], []
    with open(data_path, "r", encoding="utf-8") as fi:
        for line in fi.readlines():
            case = json.loads(line)
            gold_views = get_gold_views(case)
            views.extend(gold_views)
            if use_verifier:
                gens.extend(case["selected_facts"])
            else:
                gens.extend([random.choice(item) for item in case["pedicted_facts"]])
            c_temp = [case["legal_concept"]["criminal_circumstance"][0][0]]
            c_temp.extend([item[0] for item in case["legal_concept"]["sentence_circumstance"]])
            concepts.append(c_temp)
    bert_score = _bert_score(gens, views)
    RougeScore =   "RougeScore  | "+" | ".join([k + ":" + str(round(v['f'], 4)) for k, v in _rouge_score(gens, views).items()])
    ConceptScore = "ConceptScore| "+ str(_legal_concept_score_v2(gens, concepts))
    BleuScore =    "BleuScore   | "+" | ".join([k + " :" + str(v) for k, v in _bleu_score(gens, views).items()])
    BertScore =    "BertScore   | "+" | ".join([k + ":" + str(v) for k, v in zip(["P", "R", "F"], bert_score)])
    print(RougeScore)
    print(BleuScore)
    print(BertScore)
    print(ConceptScore)
    
def eval_solver_of_ft_based_method(data_path):
    gens, views, concepts = [], [], []
    with open(data_path, "r", encoding="utf-8") as fi:
        for line in fi.readlines():
            case = json.loads(line)
            views.append(case['output'])
            gens.append(case['response'])
    bert_score = _bert_score(gens, views)
    RougeScore =   "RougeScore  | "+" | ".join([k + ":" + str(round(v['f'], 4)) for k, v in _rouge_score(gens, views).items()])
    BleuScore =    "BleuScore   | "+" | ".join([k + " :" + str(v) for k, v in _bleu_score(gens, views).items()])
    BertScore =    "BertScore   | "+" | ".join([k + ":" + str(v) for k, v in zip(["P", "R", "F"], bert_score)])
    print(RougeScore)
    print(BleuScore)
    print(BertScore)

if __name__=="__main__":
    
    eval_solver_of_prompt_based_method(f"./outputs/lc3vg/ours/gpt3_new.json",use_verifier=True)
    # eval_solver_of_ft_based_method(f"./datasets/lc3vg/lcr_result_2.json")
    

    
