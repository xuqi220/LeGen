import os, sys
import json
from tqdm.auto import tqdm
import numpy as np
from time import time
import torch
import random
import torch.nn as nn
from utils import _load_bm25_corpus, _load_dataset, llm_infer, seg

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
setup_seed(32)

base_path = {
    "dataset":"datasets",
    "output":"outputs",
    "model":"modelfiles"
}

setting={
    'input_file_name' : "llm_cases",
    'Max_Length' : 2000,
}


def _cut_fact(text, length=1000):
    return text[:length]

def construct_prompt(cur_case, retrieved_cases=None):
    ADC_Instruction = "请根据《中华人民共和国刑法》和犯罪事实，分析犯罪情节：\n\n"
    SEC_Instruction = "请根据《中华人民共和国刑法》和犯罪事实，分析量刑情节：\n\n"
    COM_Instruction = "请根据《中华人民共和国刑法》和犯罪事实，生成法院观点（包括犯罪情节和量刑情节）：\n\n"
    
    if retrieved_cases is None: # zero shot
        fact = "".join([cur_case["bg_info"], cur_case["fact"]])[:setting["Max_Length"]]
        adc_prompt = ADC_Instruction + "犯罪事实：" + fact +"\n\n" + "本院认为："
        sec_prompt = SEC_Instruction + "犯罪事实：" + fact +"\n\n" + "本院认为："
        com_prompt = COM_Instruction + "犯罪事实：" + fact +"\n\n" + "本院认为："
    
    else:# few shot
        adc_prompt, sec_prompt, com_prompt = ADC_Instruction, SEC_Instruction, COM_Instruction
        cur_prompt = "犯罪事实：" + cur_case["bg_info"] + cur_case["fact"] + "\n\n" + "本院认为："
        for rc in retrieved_cases:
            com_list = [rc["legal_concept"]["criminal_circumstance"][0][1]]
            com_list.extend([item[1] for item in rc["legal_concept"]["sentence_circumstance"]])
            com_ = "犯罪事实：" + _cut_fact(rc["bg_info"]+rc["fact"]) + "\n\n"+"本院认为：" + "。".join(com_list)+"\n\n"

            adc_ = "犯罪事实：" + _cut_fact(rc["bg_info"]+rc["fact"]) + "\n\n"+"本院认为：" + rc["legal_concept"]["criminal_circumstance"][0][1]+"\n\n"

            sec_list = [item[1] for item in rc["legal_concept"]["sentence_circumstance"]]
            sec_ = "犯罪事实：" + _cut_fact(rc["bg_info"]+rc["fact"]) + "\n\n"+"本院认为：" + "。".join(sec_list) +"\n\n"

            adc_prompt = adc_prompt + adc_
            sec_prompt = sec_prompt + sec_
            com_prompt = com_prompt + com_

        adc_prompt = adc_prompt + cur_prompt
        sec_prompt = sec_prompt + cur_prompt
        com_prompt = com_prompt + cur_prompt
    
    return adc_prompt, sec_prompt, com_prompt

def zero_shot(llm_name):
    # load dataset
    dataset = _load_dataset(os.path.join(base_path["dataset"], f"{setting['input_file_name']}.json"))
    # prompt llm
    process_bar = tqdm(range(len(dataset)))
    res_path = os.path.join(base_path['output'],f"baselines/{setting['input_file_name']}/{llm_name}_0_shot.json")
    with open(res_path ,"w", encoding="utf-8") as fi:
        for case in dataset[:200]:
            res = {}
            # construct prompt
            adc_prompt, sec_prompt, com_prompt = construct_prompt(case)
            # prompt llm
            adc_response = llm_infer(llm_name, adc_prompt)
            sec_response = llm_infer(llm_name, sec_prompt)
            com_response = llm_infer(llm_name, com_prompt)
            # construct res
            res["id"] = case["id"]
            res["prompt"] = {"adc_prompt":adc_prompt, "sec_prompt":sec_prompt, "com_prompt":com_prompt}
            res["response"] = {"adc_response":adc_response[0], "sec_response":sec_response[0], "com_response":com_response[0]}
            res["legal_concept"] = case["legal_concept"]
            res["court_view"] = case["court_view"]
            # save res
            fi.write(json.dumps(res, ensure_ascii=False)+"\n")
            process_bar.update(1)

def few_shot(llm_name, retriever, case_num):
    # load study case
    print("loading test set")
    test_ds = _load_dataset(os.path.join(base_path["dataset"], f"{setting['input_file_name']}.json"))
    # load train set
    print("loading train set")
    train_ds = _load_dataset(os.path.join(base_path["dataset"], "train.json"))
    # prompt llm
    process_bar = tqdm(range(len(test_ds)))
    res_path = os.path.join(base_path['output'],f"baselines/{setting['input_file_name']}/{llm_name}_{case_num}_shot.json")
    with open(res_path, "w", encoding="utf-8") as fi:
        for case in test_ds: 
            res = {}
            # retrive similar case
            tokenized_query = seg.cut("".join([case["bg_info"],case["fact"]]).replace("\n", ""))
            scores = torch.from_numpy(retriever.get_scores(tokenized_query))
            top_k_indices = scores.topk(k=case_num, dim=0)[1].tolist()
            retrieved_cases = [train_ds[i] for i in top_k_indices]
            # construct prompt
            adc_prompt, sec_prompt, com_prompt = construct_prompt(case, retrieved_cases)
            # prompt llm
            adc_response = llm_infer(llm_name, adc_prompt)
            sec_response = llm_infer(llm_name, sec_prompt)
            com_response = llm_infer(llm_name, com_prompt)
            # construct res
            res["id"] = case["id"]
            res["prompt"] = {"adc_prompt":adc_prompt, "sec_prompt":sec_prompt, "com_prompt":com_prompt}
            res["response"] = {"adc_response":adc_response[0], "sec_response":sec_response[0], "com_response":com_response[0]}
            res["legal_concept"] = case["legal_concept"]
            res["court_view"] = case["court_view"]
            # save res
            fi.write(json.dumps(res, ensure_ascii=False)+"\n")
            process_bar.update(1)

def experiment_on_glm130b(retriever):
    zero_shot("glm130b")
    for num in [1,2,3,4]: 
        print(f"Mode: glm130b | Num: {num}")
        few_shot("glm130b", retriever, num)

def experiment_on_glm4(retriever):
    zero_shot("glm4")
    for num in [1,2,3,4]: 
        print(f"Mode: glm4 | Num: {num}")
        few_shot("glm4", retriever, num)

def experiment_on_gpt3(retriever):
    zero_shot("gpt3")
    for num in [3]:
        print(f"Mode: gpt3 | Num: {num}")
        few_shot("gpt3", retriever, num)

def experiment_on_gpt4(retriever):
    zero_shot("gpt4")
    for num in [1,2,3,4]:
        print(f"Mode: gpt4 | Num: {num}")
        few_shot("gpt4", retriever, num)
    

if __name__=="__main__":
    retriever = _load_bm25_corpus(os.path.join(base_path['dataset'], "tokenized_trainset.txt"))
    experiment_on_glm130b(retriever)
    # experiment_on_gpt3(retriever)
    # experiment_on_glm130b(retriever)
    
    