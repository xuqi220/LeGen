# 训练retrieval
import random
import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import classification_report, precision_score, f1_score, recall_score, accuracy_score
from torch.optim import AdamW
import torch.nn as nn
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset, Dataset
from sklearn.preprocessing import MultiLabelBinarizer
import json


SETTING = {
    "task_name": "concept_pred",
    "th":0,
    "DEVICE" : "cuda:0",
    "lr":1e-5,
    "BatchSize":32,
    "model_path" : "modelfiles/lawformer", 
    "train_data_path" : "datasets/LegalConceptReasoning/legal_concept_train.json",
    "dev_data_path" : "datasets/LegalConceptReasoning/legal_concept_val.json",
    "save_path":"outputs/models",
    "pred_res_path":"outputs/pred_res_lawformer.txt",
    "legal_concept_label_path":"datasets/LegalConceptReasoning/legal_concept_label_map.json",
    "num_epochs" : 10
}

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
setup_seed(32)

# 加载数据集
tokenizer = AutoTokenizer.from_pretrained(SETTING["model_path"])

def tokenize_function(example):
    res = tokenizer(example["facts"], max_length=1000, padding="max_length", truncation=True, return_tensors="pt")
    return res


def get_dataset():
    def trans_label(label_map, labels):
        label_map = list(label_map)
        res = []
        for label in labels:
            temp = [0]*len(label_map)
            for l in label:
                temp[label_map.index(l)] = 1
            res.append(temp)
        return res 
    # label encoder
    label_encoder = MultiLabelBinarizer()
    # load dataset
    with open(SETTING["train_data_path"], "r", encoding="utf-8") as fi:
        train_ds = json.loads(fi.read())
        label_encoder = label_encoder.fit(train_ds["labels"])
        train_ds["labels"] = trans_label(label_encoder.classes_, train_ds["labels"])
        train_ds = Dataset.from_dict(train_ds)
    with open(SETTING["dev_data_path"], "r", encoding="utf-8") as fi:
        val_ds = json.loads(fi.read())
        val_ds["labels"] = trans_label(label_encoder.classes_, val_ds["labels"])
        val_ds["facts"] = val_ds["facts"][:500]
        val_ds["labels"] = val_ds["labels"][:500]
        val_ds = Dataset.from_dict(val_ds)
    with open(SETTING["legal_concept_label_path"], "w", encoding="utf-8") as fi:
        fi.write(json.dumps(list(label_encoder.classes_), ensure_ascii=False))

    # tokenization
    tokenized_train_ds =  train_ds.map(tokenize_function, batched=True).remove_columns(["facts"])
    tokenized_val_ds =  val_ds.map(tokenize_function, batched=True).remove_columns(["facts"])
    
    # set format
    tokenized_train_ds.set_format("torch")
    tokenized_val_ds.set_format("torch")

    # dataloader
    train_dataloader = DataLoader(tokenized_train_ds, shuffle=True, batch_size=SETTING["BatchSize"])
    val_dataloader = DataLoader(tokenized_val_ds, batch_size=2*SETTING["BatchSize"])
    return train_dataloader, val_dataloader, label_encoder.classes_

class Trainer:
    def __init__(self, device, model, train_dl, val_dl, optimizer) -> None:
        self.device = device
        self.model = model.to(device)
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.optimizer = optimizer
        self.step = 1000
        self.cur_step = 0
        self.cur_min_loss = float("inf")
        self.val_loss = []

    def _run_batch(self, inputs):
        inputs = {k:v.to(self.device) for k, v in inputs.items()}
        labels = inputs.pop("labels")
        self.optimizer.zero_grad()
        outputs = self.model(**inputs)["logits"]
        loss = F.binary_cross_entropy_with_logits(outputs, labels.float())
        loss.backward()
        self.optimizer.step()
        self.process_bar.update(1)

    def _run_epoch(self):
        for batch in self.train_dl:
            self._run_batch(batch)
            if (self.cur_step+1)%self.step==0: #每1k步评估一次
                self.model.eval()
                l = self._run_eval()
                self.val_loss.append(l)
                if l<=self.cur_min_loss:
                    self._save_ckp() # save model
                    self.cur_min_loss = l
                self.model.train()
            self.cur_step+=1
    
    def train(self, max_epoch=10):
        num_training_steps = SETTING["num_epochs"] * len(self.train_dl)
        self.process_bar = tqdm(range(num_training_steps))
        for epoch in range(max_epoch):
            self._run_epoch() # train
            # torch.save(self.model, f"outputs/models/ckp_{epoch}.pkl")

    def _save_ckp(self):
        # save model
        model_name = f"ckp_{self.cur_step+1}.pkl"
        path = SETTING["save_path"] + "/" + f"{SETTING['task_name']}_{model_name}" 
        torch.save(self.model, path)
        # save loss
        # with open(SETTING["save_path"]+ "/" + "loss_list.txt", "w", encoding="utf-8") as fi:
        #     fi.write(json.dumps(self.val_loss, ensure_ascii=False))


    def _run_eval(self):
        total_loss=0
        Y = []
        Y_hat = []
        with torch.no_grad():
            for inputs in self.val_dl:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                labels = inputs.pop("labels")
                logits = self.model(**inputs)["logits"]
                loss = F.binary_cross_entropy_with_logits(logits, labels.float())
                total_loss+=loss.item()
                Y.extend(labels.tolist())
                preds = logits>SETTING["th"]
                Y_hat.extend(preds.tolist())
        acc = f"Acc: {round(accuracy_score(y_pred=Y_hat, y_true=Y), 4)} |"
        p = f"P: {round(precision_score(y_pred=Y_hat, y_true=Y, average='samples'), 4)} |"
        F1 = f"F1: {round(f1_score(y_pred=Y_hat, y_true=Y, average='samples'), 4)}"
        R = f"R: {round(recall_score(y_pred=Y_hat, y_true=Y, average='samples'), 4)} |"
        avg_loss = round(total_loss/len(self.val_dl), 4)
        records = f"Step: {self.cur_step+1} | "+ f"Loss: {avg_loss} | " + acc + p + R + F1
        print(records)
        with open(SETTING["save_path"]+ "/" + f"{SETTING['task_name']}_loss_list.txt", "a", encoding="utf-8") as fi:
            fi.write(json.dumps(records, ensure_ascii=False)+"\n")
        return avg_loss


                
def load_trainer():
    train_dl, val_dl, labels = get_dataset()
    model = AutoModelForSequenceClassification.from_pretrained(SETTING["model_path"], 
                                                               num_labels=len(labels))
    optimizer = AdamW(model.parameters(), lr=SETTING["lr"])
    # training_step = SETTING["num_epochs"] * len(train_dl)
    # scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, 
    #                                             num_warmup_steps=int(0.05*training_step),
    #                                             num_training_steps=training_step)
    # dataloader, model, optimizer scheduler = accelerator.prepare(train_dl, val_dl, model, optimizer, scheduler)
    
    trainer = Trainer(SETTING["DEVICE"], model, train_dl, val_dl, optimizer)
    return trainer
        
if __name__=="__main__":
    # inputs = tokenizer("任某提起诉讼，请求判令解除婚姻关系并对夫妻共同财产进行分割。", return_tensors="pt")
    # model = AutoModelForSequenceClassification.from_pretrained(SETTING["model_path"], 
    #                                                            num_labels=2)
    # torch.save(model, "./model.pkl")
    # loaded_model = torch.load("./model.pkl")
    # output = loaded_model(**inputs)
    trainer = load_trainer()
    trainer.train(max_epoch=SETTING["num_epochs"])

