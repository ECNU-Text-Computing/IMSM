from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType
import torch
import torch.nn as nn
import os
import json
import argparse
from torch.nn.utils.rnn import pad_sequence
from siamese_model import SiameseModel
from metric_evaluation import evaluation_all


def readlines(dataset_path):
    """ json- > List"""
    with open(dataset_path, "r", encoding="utf-8") as json_file:
        data = [json.loads(line) for line in json_file]

    return data


def model_inference(test_dataset_path, output_path, max_target_length, data_device, evaluation_datapath, siamese_model=None,
                    model_path=None, peft_path=None, mode=0, linear_A_path=None,linear_B_path=None,linear_top_A_path=None,linear_top_B_path=None):  # 加载数据

    data = readlines(test_dataset_path)
    siamese_model = siamese_model

    if model_path is not None and peft_path is not None:
        siamese_model = SiameseModel(model_path=model_path, device=data_device, data_device=data_device, peft_config=None,
                                     peft_path=peft_path, mode=mode, linear_A_path=linear_A_path,
                                     linear_B_path=linear_B_path,linear_top_A_path=linear_top_A_path,linear_top_B_path=linear_top_B_path)
    siamese_model.eval()

    dataset = output_path.split("/")[-2]
    os.makedirs(f"./dataset_out/{dataset}", exist_ok=True)

    updata_interval = 100
    process_bar = tqdm(total=len(data),
                       desc=f"Answering questions {test_dataset_path} using {peft_path} model.")  # 设置进度条
    with open(output_path, 'w', encoding="utf-8") as f_out:
        for i in range(len(data)):
            query = data[i]["input"]
            if mode == 0:
                finetune_out = siamese_model.forward(query, max_target_length)
     
            elif mode == 2 or mode == 3 or mode==4 or mode==5 or mode==6:
                finetune_out = siamese_model.forward_from_hidden_state_with_gate(query, max_target_length)

            f_out.write(
                json.dumps({"input": query, "output": data[i]['output'], "answer": finetune_out}) + "\n")



            if i % updata_interval == 0 or i == len(data):
                process_bar.update(updata_interval)

    evaluation_all(output_path, evaluation_datapath)
    print(f"evalution result saved to {evaluation_datapath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/home/llama/llama3")
    parser.add_argument("--peft_path", type=str, default=None)
    parser.add_argument("--test_dataset_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--evaluation_datapath", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:2")
    parser.add_argument("--data_device", type=str, default="cuda:2")
    parser.add_argument("--max_target_length", type=int, default=128)
    parser.add_argument("--mode", type=int, default=0)
    parser.add_argument("--linear_A_path", type=str, default=None)
    parser.add_argument("--linear_B_path", type=str, default=None)
    parser.add_argument("--linear_top_A_path", type=str, default=None)
    parser.add_argument("--linear_top_B_path", type=str, default=None)

    args = parser.parse_args()
    model_inference(args.test_dataset_path, args.output_path, args.max_target_length, args.data_device, args.evaluation_datapath,
                    siamese_model=None, model_path=args.model_path, peft_path=args.peft_path,
                    mode=args.mode, linear_A_path=args.linear_A_path, linear_B_path=args.linear_B_path,
                    linear_top_A_path=args.linear_top_A_path, linear_top_B_path=args.linear_top_B_path)

# nohup python model_inference.py --peft_path /home/xsong/SiameseModelTrain2/saved_model/adalora_webq2/baseline-qwen-4b-adalora --mode 2 --linear_path  --test_dataset_path /dataset/webq/data_test.json --output_path ./dataset_out/trainer_adalora_webq1/finetune_siamese_data_test_filter.json --evaluation_datapath ./evalution_result/adalora_boolq1_test.log --device cuda:0
