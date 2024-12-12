import argparse
import json
import os

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, AutoModel
from metric_evaluation import evaluation_all


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="/home/xsong/Qwen/Qwen1.5-4B")

    parser.add_argument("--checkpoint", type=str,
                        default="/home/xsong/SiameseTrainAAAI/baseline/lora/saved_model/GSM8K1/baseline-qwen1.5-4b-lora")

    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--evaluation_datapath", type=str)

    args = parser.parse_args()
    return args


def load_model(model_name_or_path, checkpoint):
    global model, tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

    if "llama" or "qwen" in model_name_or_path.lower():
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True).to(
            torch.cuda.current_device())
    elif "chatglm" in model_name_or_path.lower():
        model = AutoModel.from_pretrained(model_name_or_path,trust_remote_code=True).to(
            torch.cuda.current_device())

    model = PeftModel.from_pretrained(model, checkpoint)
    model = model.half()
    model = model.eval()


def readlines(dataset_path):
    """ json- > List"""
    data = []
    with open(dataset_path, 'r', encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            data.append(entry)
    return data


def generate(model_name_or_path, checkpoint, input_path, output_path, max_new_tokens, evaluation_datapath):
    load_model(model_name_or_path, checkpoint)
    os.makedirs("/".join(output_path.split("/")[:-1]), exist_ok=True)
    with open(output_path, 'w', encoding="utf-8") as f_out:
        data = readlines(input_path)

        updata_interval = 100  # 更新频率
        process_bar = tqdm(total=len(data), desc=f"Answering questions {input_path}.")  # 设置进度条
        for i in range(len(data)):
            query = data[i]["input"]
            
            ids = tokenizer.encode(query, add_special_tokens=True)

            input_ids = torch.LongTensor([ids]).to(torch.cuda.current_device())
            attention_mask = torch.ones(input_ids.shape, dtype=torch.long).to(torch.cuda.current_device())

            if "llama" or "qwen" in model_name_or_path.lower():
                generated_ids = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    
                    do_sample=False,
                    attention_mask=attention_mask,
                    pad_token_id=tokenizer.eos_token_id,
                    temperature=0

                )

                generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(input_ids, generated_ids)]
                answer = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            elif "chatglm" in model_name_or_path.lower():
                out = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,

                )
                out_text = tokenizer.decode(out[0], skip_special_tokens=True)
                answer = out_text.replace(query, "").replace("[gMASK]sop", "").strip()

            f_out.write(
                json.dumps({"input": query, "output": data[i]['output'], "answer": answer}) + "\n")

            if i % updata_interval == 0 or i == len(data):
                process_bar.update(updata_interval)
    evaluation_all(output_path, evaluation_datapath)


if __name__ == "__main__":
    args = parse_args()
    generate(args.model_name_or_path, args.checkpoint, args.input_path, args.output_path, args.max_new_tokens,
             args.evaluation_datapath)

# export CUDA_VISIBLE_DEVICES=7
# nohup python generate.py --model_name_or_path --input_path /dataset/nqopen/data_test_filter.json --output_path ./dataset_out/nqopen4_test.json --max_new_tokens 64 --checkpoint /home/xsong/SiameseModelTrain2/baseline/lora/saved_model/nqopen4/baseline-chatglm-6b-lora --evaluation_datapath
