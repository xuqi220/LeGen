# Divide and Conquer:Legal Concept-guided Criminal Court View Generation

## Introduction
In this study, we introduce a new benchmark named LCVG with annotated legal concepts. Based on the dataset, we propose a legal concept-guided court view generation framework (LeGen). Specifically, given the fact, LeGen first divides the court view into several sub-views based on the predicted legal concepts. Then the solver and verifier are employed to generate and select rationales respectively. Finally, the court view is generated by incorporating the fact and rationales. Experiments on the real word dataset demonstrate the effectiveness of our proposed method.

## Dataset
| Item | LCVG  |
|--------|--------|
| \# Train set | 60,744 | 
| \# Development set | 20,257  | 
| \# Test set | 20,290    | 
| \# Type of legal concept | 101 | 
| \# Avg. Legal Concept per case | 2.1  | 
| \# Avg. Length of fact | 781.4 | 
| \# Avg. Length of rationale | 52.8 | 
| \# Avg. Length of court view| 245.9 | 

[LeGen dataset Link](https://pan.baidu.com/s/1GsdoAVcd7KavY3Tz7SHZyA?pwd=g2zd)


## Example
```
{
    "id": 2467072, 
    "name": "司某某 
    "bg_info": "司某某，1981年2月12日出生于河xx省石家庄市无极县，初中文化，群众，务农，曾因犯交通肇事罪于2014年7月23日被无极县人民法院判处有期徒刑一年零六个月，缓刑二年。 
    "fact": "经审理查明：2017年6月6日司某某通过微信与受害人安某1相识，并称自己是无极司法局的“李某”。后司某某以其朋友需要购买减肥药为名，将受害人约至无极县七汲镇见面，... 
    "legal_concept": {
        "criminal_circumstance": [["盗窃罪 "司某某以非法占用为目的，秘密窃取他人财物，数额较大，其行为已构成盗窃罪]], 
        "sentence_circumstance": [["缓刑 "结合司某某家庭情况和河北省无极县司法局出具的《调查评估意见书》，对司某某适用缓刑不致再危害社会，对其居住地没有重大不良影响，可以对其宣告缓刑，在缓刑考验期内依法实行社区矫正]]}, 
    "court_view": "本院认为，被告人司某某以非法占用为目的，秘密窃取他人财物，数额较大，其行为已构成**盗窃罪**。公诉机关指控被告人司某某犯盗窃罪的事实清楚，证据确实充分，指控罪名成立，应当依法惩处。...结合被告人司某某家庭情况和河北省无极县司法局出具的《调查评估意见书》，对被告人司某某适用缓刑不致再危害社会，对其居住地没有重大不良影响，可以对其宣告缓刑，在缓刑考验期内依法实行社区矫正。"
}
```


## Legal Concept Distribution
The legal concept has a long-tail distribution. which impact the performance of the Decomposer module. We will explore the problem in the future work.
<img src="asset/dist.png">

## Example of Legal Concept Definition
The legal concepts are clearly defined in the Chinese Criminal Law. For example, the Article 67 define the legal concept *Voluntary Surrender*.

```
【故意犯罪】明知自己的行为会发生危害社会的结果，并且希望或者放任这种结果发生，因而构成犯罪的，是故意犯罪。故意犯罪，应当负刑事责任。
【过失犯罪】应当预见自己的行为可能发生危害社会的结果，因为疏忽大意而没有预见，或者已经预见而轻信能够避免，以致发生这种结果的，是过失犯罪。过失犯罪，法律有规定的才负刑事责任。
【自首】犯罪以后自动投案，如实供述自己的罪行的，是自首。对于自首的犯罪分子，可以从轻或者减轻处罚。其中，犯罪较轻的，可以免除处罚。被采取强制措施的犯罪嫌疑人、被告人和正在服刑的罪犯，如实供述司法机关还未掌握的本人其他罪行的，以自首论。犯罪嫌疑人虽不具有前两款规定的自首情节，但是如实供述自己罪行的，可以从轻处罚；因其如实供述自己罪行，避免特别严重后果发生的，可以减轻处罚。

```
## Framework

<img src="asset/framework.png">

## Setup
Please install all the required packages first by running the following command:

```
pip install -r requirements.txt 
```

## Quick Start

trian the decomposer
```
python decomposer_train.py
```

trian the verifier
```
python verifier_train.py
```

trian the solver
```
 CUDA_VISIBLE_DEVICES=0,1 deepspeed --master_port 29500 solver_and_generator_train.py \
                --ds_file ds_zero2_no_offload.json \
                --train_file data/solver_sft.json \
                --max_len 1560 \
                --max_src_len 1024 \
                --model_path glm3/ \
                --lora_dim 16 \
                --lora_alpha 64 \
                --lora_dropout 0.1 \
                --lora_module_name "query_key_value,dense_h_to_4h,dense_4h_to_h,dense" \
                --output_dir ./solver \
                --train_batch_size_per_device 1 \
                --gradient_accumulation_steps 4 \
                --learning_rate 1e-5 \
                --weight_decay 0.1 \
                --num_train_epoch 10 \
                --warmup_ratio 0.1 \
                --seed 2333 \
                --show_loss_step 50 \
                --save_model_step 50
```

trian the generator
```
CUDA_VISIBLE_DEVICES=0,1 deepspeed --master_port 29500 solver_and_generator_train.py \
                --ds_file ds_zero2_no_offload.json \
                --train_file data/generator_sft.json \
                --max_len 2048 \
                --max_src_len 1560 \
                --model_path glm3/ \
                --lora_dim 16 \
                --lora_alpha 64 \
                --lora_dropout 0.1 \
                --lora_module_name "query_key_value,dense_h_to_4h,dense_4h_to_h,dense" \
                --output_dir ./generator \
                --train_batch_size_per_device 1 \
                --gradient_accumulation_steps 4 \
                --learning_rate 1e-5 \
                --weight_decay 0.1 \
                --num_train_epoch 10 \
                --warmup_ratio 0.1 \
                --seed 2333 \
                --show_loss_step 50 \
                --save_model_step 50
```
