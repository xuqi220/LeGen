import random
import json, re
import os, sys
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from utils import _load_bm25_corpus, _load_dataset, llm_infer, seg, _load_article_desc, llm_load

setting = {
    "dataset":"lc3vg",
    "device":"cuda:0",
    "input_file_name":"llm_cases", # llm_cases or case_study
    "legal_concept_func":"oracle", # use legal concept in the way of oracle or prediction or retriever
    "sim_case_num":3,# sim cases for prompt selection module or gen module
    "use_SM":True, # use selection module?
    "use_verfier_in_SM":True, # verifier in selection module？
    "selection_module_sample_num":3, 
    "use_verfier_in_Gen":False, # use verifier in Generation module？
    "generation_module_sample_num":1,
    "llm_name":"gpt3",# 
    "mode":"api", # api/local
}

base_path = {
    "dataset":f"datasets/{setting['dataset']}",
    "output":f"outputs/{setting['dataset']}/ours",
    "ckp":"ckp",
    "model":"modelfiles"
}

def tokenize(tokenizer, text, max_length):
    res = tokenizer(text, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
    res = {k:v.to(setting["device"]) for k, v in res.items()}
    return res

if setting["legal_concept_func"] == "prediction":
    print("load tokenzier function of legal concept predictor...")
    legal_concept_tokenizer = AutoTokenizer.from_pretrained(f"{base_path['model']}/lawformer")
    print("load legal concept predictor...")
    legal_concept_predictor = torch.load(f"{base_path['ckp']}/concept_pred_ckp_14000.pkl")
    print("load legal concept label mapping...")
    with open(f"{base_path['dataset']}/legal_concept_label_map.json", "r", encoding="utf-8") as fi:
        legal_concept_predictor_label_map = json.loads(fi.read())
    
if setting["use_verfier_in_SM"] or setting["use_verfier_in_Gen"]:
    print("load tokenzier of verifier...")
    verifier_tokenizer = AutoTokenizer.from_pretrained(f"{base_path['model']}/roberta_wwm")
    print("load verifier...")
    verifier = torch.load(f"{base_path['ckp']}/verifier_ckp_16000.pkl")
    print("load verifier label mapping...")
    with open(f"{base_path['dataset']}/lawkg_verfier_label_map.json", "r", encoding="utf-8") as fi:
        verifier_label_map = json.loads(fi.read())

if setting["mode"]=="local": # local model
    llm_model, llm_tokenizer = llm_load(setting["llm_name"]) 

    
# legal concept prdiciton
def get_legal_concept_predictor(case):
    res = []
    fact = case['bg_info']+case["fact"]
    tokenized_fact = tokenize(tokenizer=legal_concept_tokenizer, text=fact, max_length=1000)
    logits = legal_concept_predictor(**tokenized_fact)["logits"]
    preds = logits>0
    for p, y in zip(preds.tolist()[0], legal_concept_predictor_label_map):
        if p:
            res.append(y)
    return res

# legal concept oracle
def get_legal_concept_oracle(case):
    res = [case["legal_concept"]["criminal_circumstance"][0][0]]
    for item in case["legal_concept"]["sentence_circumstance"]:
        res.append(item[0])
    return res

def _get_case_concepts(case):
    legal_concepts = [case["legal_concept"]["criminal_circumstance"][0][0]]
    for item in case["legal_concept"]["sentence_circumstance"]:
        legal_concepts.append(item[0])
    return legal_concepts

def _get_case_concept2templet(case):
    concept2templet = {}
    lc = case["legal_concept"]["criminal_circumstance"][0][0]
    templet = case["legal_concept"]["criminal_circumstance"][0][1]
    concept2templet[lc]=templet
    for item in case["legal_concept"]["sentence_circumstance"]:
        concept2templet[item[0]] = item[1]
    return concept2templet

def naive_retriever(input_case, retriever, train_dataset):
    legal_concepts = []
    top_k_cases = []
    if setting['sim_case_num']<=0:
        return top_k_cases
    # similar scores
    tokenized_query = seg.cut("".join([input_case["bg_info"],input_case["fact"]]).replace("\n", ""))
    scores = torch.from_numpy(retriever.get_scores(tokenized_query))
    # top-k cases based on fact similarity and legal concept
    top_k_indices = scores.topk(k=setting['sim_case_num'], dim=0)[1].tolist()
    for idx in top_k_indices:
        case = train_dataset[idx]
        top_k_cases.append(case)
        lcs = _get_case_concepts(case)
        legal_concepts.extend(lcs)
    return top_k_cases, list(set(legal_concepts))

# retriever 
def legal_concept_aware_retriever(input_case, legal_concepts, retriever, train_dataset):
    top_k_cases = []
    if setting['sim_case_num']<=0:
        return top_k_cases
    # similar scores
    tokenized_query = seg.cut("".join([input_case["bg_info"],input_case["fact"]]).replace("\n", ""))
    scores = torch.from_numpy(retriever.get_scores(tokenized_query))
    # top-k cases based on fact similarity and legal concept
    top_k_indices = scores.topk(k=500, dim=0)[1].tolist()
    # if use_concept:
    for idx in top_k_indices:
        if len(top_k_cases)>=setting['sim_case_num']:
            break
        case = train_dataset[idx]
        if set(_get_case_concepts(case))==set(legal_concepts):
            top_k_cases.append(case)
    return top_k_cases

def construct_selection_prompts(top_k_cases, legal_concept):
    prompt = ""
    for case in top_k_cases:
        concept2templet = _get_case_concept2templet(case)
        if legal_concept in concept2templet:
            if not concept2templet[legal_concept].endswith("。"):
                text = concept2templet[legal_concept]+"。"
            templet = f"""<犯罪事实>：{case["bg_info"]}+{case["fact"]}
            
            <{legal_concept}情节>：{text}

            """
            prompt += templet
    return prompt

def select_legal_concept_related_fact(input_case, legal_concepts, top_k_cases):
    responses = []
    for lc in legal_concepts:
        # instruction = f"你是一个法官，请分析{lc}情节\n\n"
        instruction = ""
        templets = construct_selection_prompts(top_k_cases, lc)
        input_case_prompts = f"""<犯罪事实>：{input_case["bg_info"]}+{input_case["fact"]}
        
        <{lc}情节>：
        """
        prompts = instruction+templets+input_case_prompts
        response = llm_infer(setting["llm_name"], 
                             prompt=prompts, 
                             response_num=setting["selection_module_sample_num"], 
                             stop_words=["。"])
        responses.append(response)

    # verifier the select facts
    if setting['use_verfier_in_SM']: # use verifier in selection module
        selected_facts = verfier_score(responses, legal_concepts)
    else: # not use verifier in selection module
        selected_facts = verifier_random(responses)

    return responses, selected_facts

def verifier_random(concept_related_facts,):
    facts = [random.choice(item) for item in concept_related_facts]
    return facts

def verfier_score(concept_related_facts, legal_concepts):
    selected_facts = []
    for facts, concept in zip(concept_related_facts, legal_concepts):
        tokenized_facts = tokenize(tokenizer=verifier_tokenizer, text=facts, max_length=100)
        logits = verifier(**tokenized_facts)["logits"]
        concept_logits = logits[:,verifier_label_map.index(concept)]
        concept_idx = concept_logits.argmax(dim=0)
        selected_facts.append(facts[concept_idx.item()])
    return selected_facts

def construct_cvg_prompts(top_k_cases, selected_facts):
    prompts = ""
    if selected_facts is None:
        for case in top_k_cases:
            prompts += "<犯罪事实>：" + case["fact"] + "\n\n"
            concept2templet = _get_case_concept2templet(case)
            prompts += f"<量刑情节>：{'。'.join(list(concept2templet.values()))}\n\n"
            prompts += f"<法院观点>：{case['court_view']}"
    else:
        instruction = ""
        for case in top_k_cases:
            # prompts += "【前科劣迹】：" + case["bg_info"] + "\n\n"
            # prompts += "【犯罪事实】：" + case["fact"] + "\n\n"
            concept2templet = _get_case_concept2templet(case)
            for c, t in concept2templet.items():
                prompts += f"<{c}情节>：{t}" + "\n\n"
            prompts += f"<法院观点>：{case['court_view']}"
        prompts = instruction + prompts
    return prompts

def generate_court_views(input_case, legal_concepts, top_k_cases, selected_facts):
    if selected_facts is None: # gen view based on whole fact and concepts
        prompts = construct_cvg_prompts(top_k_cases, selected_facts)
        prompts += "<犯罪事实>：" + input_case["bg_info"] + input_case["fact"] + "\n\n"
        prompts += f"请生成包含：{'，'.join(legal_concepts)}的法院观点：\n\n"
        prompts += "<法院观点>："
    else: # gen view based on selected fact only 
        prompts = construct_cvg_prompts(top_k_cases, None)
        # for c, t in zip(legal_concepts, selected_facts):
        #     prompts += f"<{c}情节>：{t}" + "\n\n"
        prompts += "<犯罪事实>：" + input_case["fact"] + "\n\n"
        prompts += f"<量刑情节>：{'。'.join(selected_facts)}\n\n"
        prompts += "<法院观点>："
    responses = llm_infer(setting["llm_name"], 
                          prompt=prompts, 
                          response_num=setting['generation_module_sample_num'])
    # verify response
    if setting["use_verfier_in_Gen"]:
        lc_sents = []
        sp_res = re.split(r"[;；。]", responses)
        for lc in legal_concepts:
            temp = []
            for s in sp_res:
                if lc in s:
                    temp.append(s)
            lc_sents.append(",".join(temp))
    else:
        res = random.choice(responses)
    return res


def reason():
    # load retriever 
    retriever = _load_bm25_corpus(data_path=f"{base_path['dataset']}/tokenized_trainset.txt")
    # load dataset
    test_datasets = _load_dataset(data_path=f"{base_path['dataset']}/{setting['input_file_name']}.json")
    train_dataset = _load_dataset(data_path=f"{base_path['dataset']}/tokenized_cases.json")
    # reasonging 
    process_bar = tqdm(range(len(test_datasets)))
    with open(f"{base_path['output']}/{setting['llm_name']}_new.json", "w", encoding="utf-8") as fi:
        for input_case in test_datasets[:200]:
            res = {}
            # get legal concepts
            if setting['legal_concept_func']=="oracle":
                legal_concepts = get_legal_concept_oracle(input_case)
                # get top k sim cases
                top_k_cases = legal_concept_aware_retriever(input_case, legal_concepts, retriever, train_dataset)
            elif setting['legal_concept_func']=="lawformer":
                legal_concepts = get_legal_concept_predictor(input_case)
                # get top k sim cases
                top_k_cases = legal_concept_aware_retriever(input_case, legal_concepts, retriever, train_dataset)
            else: # naive retriever
                top_k_cases, legal_concepts = naive_retriever(input_case, retriever, train_dataset)
            selected_facts = None
            if setting['use_SM']: # use selection module
                # select legal concept related facts
                predicted_facts, selected_facts = select_legal_concept_related_fact(input_case, legal_concepts, top_k_cases)
                # genenrate court views
                cvs = generate_court_views(input_case, legal_concepts, top_k_cases, selected_facts)
            else: # not use selection module
                # gen views directlyNone
                cvs = generate_court_views(input_case, legal_concepts, top_k_cases, selected_facts=None)
            # construct res
            res["id"] = input_case["id"]
            res['bg_info']=input_case['bg_info']
            res['fact'] = input_case['fact']
            res["legal_concepts_"] = legal_concepts
            res["selected_facts"] = selected_facts
            res["pedicted_facts"] = predicted_facts
            res["response"] = cvs
            res["legal_concept"] = input_case["legal_concept"]
            res["court_view"] = input_case["court_view"]
            # save res
            fi.write(json.dumps(res, ensure_ascii=False)+"\n")
            process_bar.update(1)


if __name__=="__main__":
    reason()
