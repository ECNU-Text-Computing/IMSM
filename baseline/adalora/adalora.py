import argparse
import json
import os
import torch
torch.set_num_threads(16)
#from torch.utils.data import Dataset
from datasets import Dataset
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
    Seq2SeqTrainer,
    TrainingArguments,
    Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, Trainer
)
from peft import (
    TaskType,
    AdaLoraConfig,
    get_peft_model,
    set_peft_model_state_dict,
    prepare_model_for_int8_training
)
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '/SiameseTrainAAAI/baseline'))
sys.path.append(project_root)

from generate import *

def parse_args():
    parser = argparse.ArgumentParser(description="AdaLoRA")
    parser.add_argument("--train_args_file", type=str, default="adalora.json")
    parser.add_argument("--model_name_or_path", type=str, default="/ChatGLM3/chatglm3-6b")
    parser.add_argument("--data_path", type=str, default="/dataset/ropes/data_train.json")
    parser.add_argument("--test_path", type=str,
                        default="/dataset/ropes/data_test.json")
    parser.add_argument("--output_path", type=str,
                        default="/dataset_out/qwen_ropes_2024.json")
    parser.add_argument("--evaluation_datapath", type=str,
                        default="/evaluation_result/qwen_ropes_2024.log")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_input_length", type=int, default=1024)
    parser.add_argument("--max_output_length", type=int, default=16)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--int8", action="store_true")
    parser.add_argument("--no_gradient_checkpointing", action="store_true")
    args = parser.parse_args()
    return args


class DataCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        lengths = [len(feature["input_ids"]) for feature in batch]
        longest = max(lengths)
        input_ids,attention_mask, labels = [], [],[]
        for length, feature in sorted(zip(lengths, batch), key=lambda x: -x[0]):
            pad_len = longest - length
            ids = feature["input_ids"] + [self.pad_token_id] * pad_len
            mask = feature["attention_mask"] + [0] * pad_len
            label = feature["labels"] + [-100] * pad_len
            input_ids.append(torch.LongTensor(ids))
            attention_mask.append(torch.LongTensor(mask))
            labels.append(torch.LongTensor(label))

        input_ids = torch.stack(input_ids)
        attention_mask = torch.stack(attention_mask)
        labels = torch.stack(labels)

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def load_custom_dataset(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return Dataset.from_dict({key: [example[key] for example in data] for key in data[0].keys()})

def tokenize_function(example, tokenizer, max_input_length, max_output_length):
    question = example["input"]
    answer = example["output"]

    q_ids = tokenizer(text=question, add_special_tokens=True, max_length=max_input_length,
                      truncation=True)
    a_ids = tokenizer(text=answer, add_special_tokens=False, max_length=max_output_length - 1,
                      truncation=True)

    question_length = len(q_ids["input_ids"])
    input_ids = q_ids["input_ids"] + a_ids["input_ids"] + [tokenizer.eos_token_id]
    attention_mask = q_ids["attention_mask"] + a_ids["attention_mask"] + [1]
    labels = [-100] * question_length + a_ids["input_ids"] + [tokenizer.eos_token_id]

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def load_dataset(data_path, tokenizer, max_input_length, max_output_length):
    custom_dataset = load_custom_dataset(data_path)
    tokenized_dataset = custom_dataset.map(lambda x: tokenize_function(x, tokenizer, max_input_length, max_output_length))
    return tokenized_dataset

class ModifiedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        return model(input_ids=inputs["input_ids"],labels=inputs["labels"]).loss



def train(args):
    parser = HfArgumentParser(TrainingArguments)
    training_args, = parser.parse_json_file(json_file=args.train_args_file)
    print(training_args)


    # Set seed
    set_seed(args.seed)
    training_args.seed = args.seed

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token_id == None:
        pad_token_id = tokenizer.eos_token_id
    else:
        pad_token_id = tokenizer.pad_token_id

    model = None
    # Load model
    if args.int8:
        if "llama" or "qwen" in args.model_name_or_path.lower():
            model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, load_in_8bit=True,
                                                         device_map={'': torch.cuda.current_device()},
                                                         trust_remote_code=True)
        elif "chatglm" in args.model_name_or_path.lower():
            model = AutoModel.from_pretrained(args.model_name_or_path, load_in_8bit=True,
                                              device_map={'': torch.cuda.current_device()}, trust_remote_code=True)

    else:
        if "llama" or "qwen" in args.model_name_or_path.lower():
            model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, trust_remote_code=True)
            model = model.half()
        elif "chatglm" in args.model_name_or_path.lower():
            model = AutoModel.from_pretrained(args.model_name_or_path, trust_remote_code=True)
            model = model.half()

    model.config.use_cache = False
    if not args.no_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()


    # Define adaLoRA Config
    # Define target module
    target_modules = None
    if "llama" in args.model_name_or_path.lower():
        target_modules = ['q_proj', 'v_proj']
    elif "chatglm" in args.model_name_or_path.lower():
        target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING["chatglm"]
    elif "qwen" in args.model_name_or_path.lower():
        target_modules = ['q_proj']


    adaLoRA_config = AdaLoraConfig(
        peft_type="ADALORA",
        task_type="CAUSAL_LM",
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        inference_mode=False,
    )


    # add adaLoRA adaptor
    model = get_peft_model(model, adaLoRA_config)

    resume_from_checkpoint = args.resume_from_checkpoint
    if resume_from_checkpoint is not None:
        # Full checkpoint
        checkpoint_name = os.path.join(resume_from_checkpoint, "pytorch_model.bin")
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only adaLoRA model - adaLoRA config above has to fit
            resume_from_checkpoint = False  # So the trainer won't try loading its state
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()

    # Load dataset
    train_dataset = load_dataset(args.data_path, tokenizer, args.max_input_length, args.max_output_length)


    # trainer
    trainer = ModifiedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollator(pad_token_id=pad_token_id)
    )
    # train model
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    # Save our AdaLoRA model & tokenizer results
    trainer.model.save_pretrained(training_args.output_dir)
    # tokenizer.save_pretrained(training_args.output_dir)
    generate(args.model_name_or_path, training_args.output_dir, args.test_path, args.output_path, args.max_output_length, args.evaluation_datapath)

if __name__ == "__main__":
    args = parse_args()
    print(args)
    train(args)

