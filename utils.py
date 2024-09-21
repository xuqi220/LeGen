import os
import json
import torch
import random
import numpy as np
from transformers import set_seed
from torch.utils.data import Dataset, DataLoader
from modelfiles.glm3.tokenization_chatglm import ChatGLMTokenizer
from zhipuai import ZhipuAI
from openai import OpenAI
import json
from rank_bm25 import BM25Okapi
import pkuseg
seg = pkuseg.pkuseg(user_dict=["<被告人A>", "被告人"])
from personal_setting import zhipu_api_key, openai_api_key

openai_client = OpenAI(api_key=openai_api_key)
zp_client = ZhipuAI(api_key=zhipu_api_key)


def _load_article_desc(data_path):
    with open(data_path, "r", encoding="utf-8") as fi:
        art2desc = json.loads(fi.read())
    return art2desc

def _load_dataset(data_path):
    data = []
    with open(data_path, "r", encoding="utf-8") as fi:
        for line in fi.readlines():
            data.append(json.loads(line))
    return data

def _load_bm25_corpus(data_path="datasets/tokenized_trainset.txt"):
    print("loading tokenized_trainset...")
    train_ds = _load_dataset(data_path)
    print("construct bm25 corpus ...")
    bm25_corpus = BM25Okapi(train_ds)
    return bm25_corpus

def llm_load(llm_name):
    pass

def _invoke_glm130b(input_content, response_num, stop_words=None):
    responses = []
    for _ in range(response_num):
        completion = zp_client.chat.completions.create(
            model="glm-3-turbo",  
            messages=[
                {"role": "user", "content": input_content},
            ],
            temperature=0.8,
            stop=stop_words,
        )
        response = completion.choices[0].message.content
        responses.append(response)
    return responses

def _invoke_glm4(input_content, response_num, stop_words=None):
    responses = []
    for _ in range(response_num):
        completion = zp_client.chat.completions.create(
            model="glm-4",  
            messages=[
                {"role": "user", "content": input_content},
            ],
        temperature=0.5,
        stop=stop_words,
        )
        response = completion.choices[0].message.content
        responses.append(response)
    return responses

def _invoke_gpt3(input_content, response_num, stop_words=None):
    completion = openai_client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": input_content}
        ],
        temperature=1.0,
        stop=stop_words,
        n=response_num,
    )
    responses = []
    for choice in completion.choices:
        responses.append(choice.message.content)
    return responses

def _invoke_gpt4(input_content, response_num, stop_words=None):
    completion = openai_client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "user","content": input_content}
        ],
        temperature=1.0,
        stop=stop_words,
        n=response_num,
    )
    responses = []
    for choice in completion.choices:
        responses.append(choice.message.content)
    return responses

def llm_infer(llm_name, prompt, response_num=1, stop_words=None):
    assert llm_name in ["glm130b", "gpt3", "gpt4", "glm4"]
    try:
        if llm_name == "glm130b":
            response = _invoke_glm130b(prompt, response_num, stop_words)
        if llm_name == "glm4":
            response = _invoke_glm4(prompt, response_num, stop_words)
        if llm_name == "gpt3":
            response = _invoke_gpt3(prompt, response_num, stop_words) 
        if llm_name == "gpt4":
            response = _invoke_gpt4(prompt, response_num, stop_words)
    except:
        response = ["##出现错误##"]
    return response
    
def set_random_seed(seed):
    if seed is not None:
        set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def print_rank_0(msg, rank=0):
    if rank <= 0:
        print(msg)
        
def to_device(batch, device):
    output = {}
    for k, v in batch.items():
        try:
            output[k] = v.to(device)
        except:
            output[k] = v
    return output
        
def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    print("trainable params: {} || all params: {} || trainable%: {}".format(trainable_params, all_param,
                                                                            100 * trainable_params / all_param))

def save_model(model, tokenizer, output_dir, model_name, state_dict=None):
    save_dir = os.path.join(output_dir, model_name)
    if state_dict == None:
        model.save_pretrained(save_dir, torch_dtype=torch.float16)
    else:
        model.save_pretrained(save_dir, state_dict=state_dict, torch_dtype=torch.float16)
    tokenizer.save_pretrained(save_dir)
    

class GLMPromptDataSet(Dataset):
    def __init__(self, data_path, tokenizer, max_len, max_src_len, is_skip):
        self.all_data = []
        skip_data_number = 0
        with open(data_path, "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                sample = json.loads(line.strip())
                skip_flag = False
                t1 = [tokenizer.get_command("<|user|>")]
                t2 = tokenizer.encode("\n", add_special_tokens=False)
                t3 = tokenizer.encode(sample["instruction"] + sample["input"], add_special_tokens=False)
                src_tokens =  t1 + t2 + t3
                             
                if len(src_tokens) > max_src_len:
                    # 当输入内容超长时，随向后截断
                    src_tokens = src_tokens[:max_src_len]
                    skip_flag = True

                max_tgt_len = max_len - 6 - len(src_tokens)
                tgt_tokens = [tokenizer.get_command("<|assistant|>")] + tokenizer.encode("\n", add_special_tokens=False) + \
                             tokenizer.encode(sample["output"], add_special_tokens=False)

                if len(tgt_tokens) > max_tgt_len:
                    # 当输出内容超长时，随向后截断
                    tgt_tokens = tgt_tokens[:max_tgt_len]
                    skip_flag = True

                # ChatGLM3需要增加[gMASK]、sop两个标记
                t4 = [tokenizer.get_command("[gMASK]"), tokenizer.get_command("sop")]
                input_ids =  t4 + src_tokens + tgt_tokens + [tokenizer.eos_token_id]
                
                context_length = len(src_tokens) + 2
                labels = [-100] * context_length + input_ids[context_length:]

                assert len(input_ids) == len(labels)
                assert len(input_ids) <= max_len
                if is_skip and skip_flag:
                    skip_data_number += 1
                    continue
                self.all_data.append({"input_ids": input_ids, "labels": labels})
        print("the number of skipping data is {}".format(skip_data_number))

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, item):
        instance = self.all_data[item]
        return instance
    

class DataCollator(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, batch):
        lengths = [len(instance["input_ids"]) for instance in batch]
        batch_max_len = max(lengths)

        input_ids_batch, labels_batch = [], []
        for instance in batch:
            input_ids = instance["input_ids"]
            labels = instance["labels"]

            padding_len = batch_max_len - len(input_ids)
            input_ids = input_ids + [self.pad_token_id] * padding_len
            labels = labels + [-100] * padding_len

            input_ids_batch.append(input_ids)
            labels_batch.append(labels)

        return {"input_ids": torch.tensor(input_ids_batch, dtype=torch.long),
                "labels": torch.tensor(labels_batch, dtype=torch.long)}
