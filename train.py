import torch

torch.cuda.init()
from torch.cuda.amp import autocast, GradScaler

import random
import numpy as np
import torch

# set seed
seed_value = 2024
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_num_threads(16)


import json
import argparse
import os

import torch
import torch_optimizer
from peft import LoraConfig, AdaLoraConfig, IA3Config, TaskType
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from siamese_model import SiameseModel
from model_inference import model_inference
from load_dataset import *
from itertools import chain

# train
def train(siamese_model, dataset, dataloader, test_data_path, max_output_length, loss_fun, optimizer,
          accumulation_steps, lr_schedule, num_epochs, data_device, mode=0):
    scaler = GradScaler()
    for epoch in range(num_epochs):

        siamese_model.train()
        total_loss = 0
        num_batches = len(dataloader)

        for step, batch in enumerate(t := tqdm(dataloader)):

            input_ids = batch["input_ids"]  # [batch, max_len]
            label = batch["labels"]  # [batch, max_len]
            attention_mask = batch["attention_mask"]
            query_length = batch["question_lengths"]
            # calculate logits
            logits_tensor = None
            with autocast():
                if mode == 0:
                    logits_tensor = siamese_model.cal_logits(input_ids,attention_mask).to(data_device)  # [batch, max_len,vocab_size]
       
                elif mode == 2 or mode==3 or mode==4 or mode==5 or mode==6:
                    logits_tensor = siamese_model.cal_logits_from_hidden_state_with_gate(input_ids,query_length).to(data_device)


            logits_flat = logits_tensor.view(-1, logits_tensor.size(-1)).to(
                data_device)  # [batch * max_len, vocab_size]
            label_flat = label.view(-1).to(data_device)  # [batch * max_len]
 
            # loss
            with autocast():
                loss = loss_fun(logits_flat, label_flat, ignore_index=-100).to(data_device)
                loss = loss / accumulation_steps
            scaler.scale(loss).backward()

            if (step + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                #optimizer.step()  # optimizer the net
                optimizer.zero_grad()  # reset grdient

                if lr_schedule is not None:
                    lr_schedule.step()


            total_loss += loss.detach().item() * accumulation_steps
            t.set_description(f"loss: {loss * accumulation_steps}")

            if (step + 1) % 50 == 0:
                print(f'Step {step + 1}/{len(dataloader)}, Loss: {loss.item() * accumulation_steps}')


        avg_loss = total_loss / num_batches
        print(f"train_epoch:{epoch}\tavg_loss:{avg_loss}")


        os.makedirs(f"./saved_model/{dataset}", exist_ok=True)
        peft_model_id = f"./saved_model/{dataset}/{epoch}_{peft_config.peft_type}_{peft_config.task_type}"
        siamese_model.tuned_model.save_pretrained(peft_model_id)
        print(f"{epoch} model saved to {peft_model_id}")

        if mode in [2,3,4,5]:
            linear_path_A = f"./saved_model/{dataset}/{epoch}_linear_A.pth"
            torch.save(siamese_model.linear_A, linear_path_A)

            linear_path_B = f"./saved_model/{dataset}/{epoch}_linear_B.pth"
            torch.save(siamese_model.linear_B, linear_path_B)
            if mode == 3:
                linear_top_A_path = f"./saved_model/{dataset}/{epoch}_linear_top_A.pth"
                torch.save(siamese_model.linear_top_A, linear_top_A_path)

                linear_top_B_path = f"./saved_model/{dataset}/{epoch}_linear_top_B.pth"
                torch.save(siamese_model.linear_top_B, linear_top_B_path)


        if "gsm" in test_data_path.lower():
            if epoch >= num_epochs-1:
                if test_data_path is not None:
                    test(siamese_model, dataset, test_data_path, max_output_length, data_device, epoch, mode)
        else:
            if epoch >= 0:
                if test_data_path is not None:
                    test(siamese_model, dataset, test_data_path, max_output_length, data_device, epoch, mode)


def test(siamese_model, dataset, test_data_path,max_output_length, data_device, epoch, mode):

    os.makedirs(f"/dataset_out/{dataset}",exist_ok=True)
    output_path = f"/dataset_out/{dataset}/data_test_{epoch}.json"
    evaluation_datapath = f"/evaluation_result/{dataset}_test_{epoch}.log"

    siamese_model.eval()

    model_inference(test_data_path, output_path, max_output_length, data_device,
                    evaluation_datapath,
                    siamese_model=siamese_model, model_path=None, peft_path=None,
                    mode=mode, linear_A_path=None, linear_B_path=None,
                    linear_top_A_path=None, linear_top_B_path=None)


    print(f"train_epoch:{epoch}\tevaluation_result_saved_to:{evaluation_datapath}")


if __name__ == "__main__":
    '''
    step1: hyper-parameter
    '''
    config_file = "config.json"
    with open(config_file) as f:
        config = json.load(f)

    parser = argparse.ArgumentParser()
    for key, value in config.items():
        parser.add_argument(f"--{key}", default=value, type=type(value))
    args = parser.parse_args()

    '''
    step2: load model
    '''

    peft_config = None

    target_modules = None
    feedforward_modules = None
    if "llama-2" in args.model_path.lower():
        if args.peft == "lora" or args.peft=="adalora":
            target_modules = ["gate_proj","down_proj","up_proj"]
            print("llama")
    elif "llama" in args.model_path.lower():
        if args.peft == "lora" or args.peft=="adalora":
            target_modules = ['q_proj', 'v_proj']
        elif args.peft == "ia3":
            target_modules = ["k_proj", "v_proj", "down_proj"]
            feedforward_modules = ["down_proj"]
    elif "chatglm" in args.model_path.lower():
        if args.peft == "lora" or args.peft=="adalora":
            target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING["chatglm"]
        elif args.peft == "ia3":
            target_modules = ["query_key_value", "mlp.dense_4h_to_h"]
            feedforward_modules = ["mlp.dense_4h_to_h"]
    elif "qwen" in args.model_path.lower():
        if args.peft == "lora" or args.peft=="adalora":
            target_modules = ['q_proj']
        elif args.peft == "ia3":
            target_modules = ['q_proj', 'v_proj']
            feedforward_modules = []

    if args.peft == "lora":
        peft_config = LoraConfig(task_type="CAUSAL_LM",
                                 inference_mode=False,
                                 r=args.lora_r,
                                 lora_alpha=args.lora_alpha,
                                 lora_dropout=args.lora_dropout,
                                 bias="none",
                                 target_modules=target_modules)
    elif args.peft == "dora": # default target
        peft_config = LoraConfig(task_type="CAUSAL_LM",
                                 inference_mode=False,
                                 r=args.lora_r,
                                 lora_alpha=args.lora_alpha,
                                 lora_dropout=args.lora_dropout,
                                 bias="none",
                                 use_dora=True)
    elif args.peft == "adalora":
        peft_config = AdaLoraConfig(
            peft_type="ADALORA",
            task_type="CAUSAL_LM",
            inference_mode=False,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
        )
    elif args.peft == "ia3":
        peft_config = IA3Config(task_type=TaskType.CAUSAL_LM,
                                target_modules=target_modules,
                                inference_mode=False,
                                feedforward_modules=feedforward_modules)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    siamese_model = SiameseModel(args.model_path, args.device,args.data_device, peft_config=peft_config,mode=args.mode,gate_rank=args.gate_rank,top_rank=args.top_rank,dropout_prob=args.dropout_prob)

    '''
    step3: load data
    '''
    if tokenizer.pad_token_id:
        pad_token_id = tokenizer.pad_token_id
    else:
        pad_token_id = tokenizer.eos_token_id
    train_dataset = load_dataset(args.data_path, tokenizer, args.max_input_length, args.max_output_length)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, pin_memory=True,
                                  collate_fn=DataCollator(pad_token_id=pad_token_id))

    params = filter(lambda p: p.requires_grad, siamese_model.tuned_model.parameters())

    loss_fun = F.cross_entropy

    optimizer = None
    if args.optimizer == "adamw":
        if args.mode == 2 or args.mode==4 or args.mode==5 or args.mode==6:
            optimizer = torch.optim.AdamW([
                {'params': params, 'lr': args.learning_rate, },
                {'params': siamese_model.linear_A.parameters()},
                {'params': siamese_model.linear_B.parameters()},
            ],weight_decay=args.weight_decay)
        elif args.mode == 3:
            optimizer = torch.optim.AdamW([
                {'params': params, 'lr': args.learning_rate, },
                {'params': siamese_model.linear_A.parameters()},
                {'params': siamese_model.linear_B.parameters()},
                {'params': siamese_model.linear_top_A.parameters()},
                {'params': siamese_model.linear_top_B.parameters()},
            ], weight_decay=args.weight_decay)
        elif args.mode==0:
            optimizer = torch.optim.AdamW(params, lr=args.learning_rate,weight_decay=args.weight_decay)

    elif args.optimizer == "adafactor":
        if args.mode == 2 or args.mode == 4 or args.mode==5 or args.mode==6:

            params = filter(lambda p: p.requires_grad, siamese_model.tuned_model.parameters())
            params_A = siamese_model.linear_A.parameters()
            params_B = siamese_model.linear_B.parameters()

            all_params = chain(params, params_A, params_B)
            optimizer = torch_optimizer.Adafactor(all_params, lr=args.learning_rate, weight_decay=args.weight_decay)
            #optimizer = torch_optimizer.Adafactor([params, siamese_model.linear_A.parameters(),siamese_model.linear_B.parameters()], lr=args.learning_rate, weight_decay=1e-05)
        elif args.mode == 3:

            params = filter(lambda p: p.requires_grad, siamese_model.tuned_model.parameters())
            params_A = siamese_model.linear_A.parameters()
            params_B = siamese_model.linear_B.parameters()
            params_top_A = siamese_model.linear_top_A.parameters()
            params_top_B = siamese_model.linear_top_B.parameters()

            all_params = chain(params, params_A, params_B,params_top_A,params_top_B)
            optimizer = torch_optimizer.Adafactor(all_params, lr=args.learning_rate, weight_decay=args.weight_decay)

        elif args.mode == 0:
            optimizer = torch_optimizer.Adafactor(params, lr=args.learning_rate,weight_decay=args.weight_decay)

    lr_schedule = None
    if args.lr_schedule == "cosine":
        total_steps = int(len(train_dataloader) * args.num_epochs / args.accumulation_steps)
        num_warmup_steps = int(total_steps * args.num_warmup_ratio)
        lr_schedule = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,num_training_steps=total_steps)

    elif args.lr_schedule == "linear":
        total_steps = int(len(train_dataloader) * args.num_epochs / args.accumulation_steps)
        lr_schedule = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=int(args.num_warmup_ratio * total_steps),
            num_training_steps=total_steps
        )

    print(args)
    print(seed_value)
    train(siamese_model, args.dataset, train_dataloader, args.test_data_path, args.max_output_length, loss_fun,
          optimizer, args.accumulation_steps, lr_schedule, args.num_epochs, args.data_device, args.mode)
# nohup python train.py --dataset lora_webq1  --data_path /dataset/webq/data_train_filter.json --max_input_length 128 --max_output_length 64 --batch_size 16 --device cuda:3 --data_device cuda:1 --optimizer adamw --num_epochs 10 --lr_schedule None --peft lora > lora_webq1.log &
