import random
import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import classification_report, precision_score, f1_score, recall_score
from torch.optim import AdamW
import torch.nn as nn
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset, Dataset
from sklearn.preprocessing import MultiLabelBinarizer
import json
from accelerate import Accelerator
accelerator = Accelerator()

SETTING = {
    "DEVICE" : "cuda:0",
    "lr":1e-5,
    "BatchSize":32,
    "model_path" : "modelfile/roberta_wwm", 
    "train_data_path" : "datasets/LegalConceptReasoning/lawkg_verfier_train.json",
    "dev_data_path" : "datasets/LegalConceptReasoning/lawkg_verfier_val.json",
    "save_path":"outputs/models",
    "pred_res_path":"outputs/pred_res_lawformer.txt",
    "lawkg_verfier_label_path":"datasets/LegalConceptReasoning/lawkg_verfier_label_map.json",
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
    res = tokenizer(example["input_texts"], max_length=100, padding="max_length", truncation=True, return_tensors="pt")
    return res


def get_dataset():
    # load dataset
    with open(SETTING["train_data_path"], "r", encoding="utf-8") as fi:
        train_ds = Dataset.from_dict(json.loads(fi.read()))
    with open(SETTING["dev_data_path"], "r", encoding="utf-8") as fi:
        d = json.loads(fi.read())
        # d["input_texts"] = d["input_texts"][:6000]
        # d["labels"] = d["labels"][:6000]
        val_ds = Dataset.from_dict(d)

    # encode column
    train_ds = train_ds.class_encode_column("labels")
    val_ds = val_ds.class_encode_column("labels")
    with open(SETTING["lawkg_verfier_label_path"], "w", encoding="utf-8") as fi:
        fi.write(json.dumps(train_ds.features["labels"].names, ensure_ascii=False))

    # tokenization
    tokenized_train_ds =  train_ds.map(tokenize_function, batched=True).remove_columns(["input_texts"])
    tokenized_val_ds =  val_ds.map(tokenize_function, batched=True).remove_columns(["input_texts"])
    
    # set format
    tokenized_train_ds.set_format("torch")
    tokenized_val_ds.set_format("torch")

    # dataloader
    train_dataloader = DataLoader(tokenized_train_ds, shuffle=True, batch_size=SETTING["BatchSize"])
    val_dataloader = DataLoader(tokenized_val_ds, batch_size=2*SETTING["BatchSize"])
    return train_dataloader, val_dataloader, train_ds.features["labels"].names

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
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        self.optimizer.step()
        self.process_bar.update(1)

    def _run_epoch(self):
        for batch in self.train_dl:
            self._run_batch(batch)
            if (self.cur_step+1)% self.step==0: #每1k步评估一次
                self.model.eval()
                l = self._run_eval()
                self.val_loss.append(l)
                if l<self.cur_min_loss:
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
        path = SETTING["save_path"] + "/" + model_name
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
                outputs = self.model(**inputs)["logits"]
                loss = F.cross_entropy(outputs, labels)
                total_loss+=loss.item()
                Y.extend(labels.tolist())
                preds = torch.argmax(outputs, dim=-1)
                Y_hat.extend(preds.tolist())
        p = f"P : {round(precision_score(y_pred=Y_hat, y_true=Y, average='macro'), 4)}"
        r = f"R : {round(recall_score(y_pred=Y_hat, y_true=Y, average='macro'), 4)}"
        f = f"F1: {round(f1_score(y_pred=Y_hat, y_true=Y, average='macro'), 4)}"
        avg_loss = round(total_loss/len(self.val_dl), 4)
        records = f"Step: {self.cur_step+1} | "+ f"Loss: {avg_loss} | " + p + " | " + r + " | " + f +"\n"
        with open(SETTING["save_path"]+ "/" + "loss_list.txt", "a", encoding="utf-8") as fi:
            fi.write(json.dumps(records, ensure_ascii=False))
        return avg_loss


                
def load_trainer():
    train_dl, val_dl, label_encoder = get_dataset()
    model = AutoModelForSequenceClassification.from_pretrained(SETTING["model_path"], 
                                                               num_labels=len(label_encoder))
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

