import torch
import json
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from datasets import Dataset

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

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "question_length": question_length}


def load_dataset(data_path, tokenizer, max_input_length, max_output_length):
    custom_dataset = load_custom_dataset(data_path)
    tokenized_dataset = custom_dataset.map(
        lambda x: tokenize_function(x, tokenizer, max_input_length, max_output_length))
    return tokenized_dataset


class DataCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        lengths = [len(feature["input_ids"]) for feature in batch]
        longest = max(lengths)
        input_ids, attention_mask, labels, question_lengths = [], [], [],[]
        for length, feature in sorted(zip(lengths, batch), key=lambda x: -x[0]):
            pad_len = longest - length
            ids = feature["input_ids"] + [self.pad_token_id] * pad_len

            mask = feature["attention_mask"] + [0] * pad_len
            label = feature["labels"] + [-100] * pad_len
            question_length = feature["question_length"]
            input_ids.append(torch.LongTensor(ids))
            attention_mask.append(torch.LongTensor(mask))
            labels.append(torch.LongTensor(label))
            question_lengths.append(question_length)

        input_ids = torch.stack(input_ids)
        attention_mask = torch.stack(attention_mask)
        labels = torch.stack(labels)
        question_lengths = torch.tensor(question_lengths)

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "question_lengths":question_lengths}

