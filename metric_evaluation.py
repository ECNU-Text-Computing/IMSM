import string
import regex
import json
import argparse
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.metrics import f1_score, matthews_corrcoef
from rouge_score import rouge_scorer
from statistics import mean
import os 
def readlines(path):
    """Read JSON data -> List"""
    if "loramoe" in path.lower():
        with open(path, 'r', encoding='utf-8') as file:
            data = json.load(file)
    else:
        with open(path, "r", encoding="utf-8") as json_file:
            data = [json.loads(line) for line in json_file]
    return data


def normalize_answer(s):
    """Normalize answers for comparison"""

    def remove_articles(text):
        return regex.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def evaluate_em(golden_answers, generated_answers):
    em_scores = []
    for golden_answer, generated_answer in zip(golden_answers, generated_answers):
        if isinstance(golden_answer, list):
            score = 0
            generated_answer = normalize_answer(generated_answer)
            for golden in golden_answer:
                golden = normalize_answer(golden)
                score = max(score, 1 if golden == generated_answer else 0)
            em_scores.append(score)
        else:        
            golden_answer = normalize_answer(golden_answer)
            generated_answer = normalize_answer(generated_answer)
            em_scores.append(1 if golden_answer == generated_answer else 0)
    return {"EM":mean(em_scores),"Data Length":len(em_scores)}
    
    
def evaluate_contain(golden_answers, generated_answers):
    contain_scores = []
    for golden_answer, generated_answer in zip(golden_answers, generated_answers):
        if isinstance(golden_answer, list):
            score = 0
            generated_answer = normalize_answer(generated_answer)
            for golden in golden_answer:
                golden = normalize_answer(golden)
                score = max(score, 1 if golden in generated_answer else 0)
            contain_scores.append(score)
        else:  
            golden_answer = normalize_answer(golden_answer)
            generated_answer = normalize_answer(generated_answer)
            contain_scores.append(1 if golden_answer in generated_answer else 0)
    return {"Contain":mean(contain_scores),"Data Length":len(contain_scores)}
    


def evaluate_f1_overlap(golden_answers, generated_answers):
    f1_scores = []
    for golden, generated in zip(golden_answers, generated_answers):
        golden_tokens = set(normalize_answer(golden).split())
        generated_tokens = set(normalize_answer(generated).split())
        overlapping_tokens = golden_tokens.intersection(generated_tokens)
        precision = len(overlapping_tokens) / len(generated_tokens) if generated_tokens else 0
        recall = len(overlapping_tokens) / len(golden_tokens) if golden_tokens else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0
        f1_scores.append(f1)
    return {"F1_overlap":mean(f1_scores),"Data Length":len(f1_scores)}


def evaluate_mcc(golden_answers, generated_answers):
    y_true = [1 if "yes" in golden.lower() else 0 for golden in golden_answers]
    y_pred = [1 if "yes" in generated.lower() else 0 for generated in generated_answers]
    return {"MCC":matthews_corrcoef(y_true, y_pred),"Data Length":len(y_pred)}

def evaluate_f1(golden_answers, generated_answers):
    y_true = [1 if "yes" in golden.lower() else 0 for golden in golden_answers]
    y_pred = [1 if "yes" in generated.lower() else 0 for generated in generated_answers]
    return {"F1_score":f1_score(y_true, y_pred),"Data Length":len(y_pred)}


def evaluate_rouge_scores(golden_answers, generated_answers):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    for golden, generated in zip(golden_answers, generated_answers):
        scores = scorer.score(golden, generated)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)
    return {"Rouge1":mean(rouge1_scores), "Rouge2":mean(rouge2_scores), "RougeL":mean(rougeL_scores),"Data Length":len(rouge1_scores)}


def evaluation_all(datapath, evaluation_datapath):
    data = readlines(datapath)
    if "target" in data[0] and "loramoe" in datapath.lower():
        
        golden_answers = [item['target'] for item in data]

        generated_answers = [item['output'] for item in data]
    else:
        golden_answers = [item['output'] for item in data]
        generated_answers = [item['answer'] if "answer" in item else item["chatglm_answer"] for item in data]

    result = None
    
    if "cf" in datapath:
        test_name = os.path.basename(datapath)
    else:
        test_name = datapath
    if "mrpc" in test_name.lower() or "multirc" in test_name.lower() or "boolq" in test_name.lower():
        em_result = evaluate_em(golden_answers, generated_answers)
        f1_result = evaluate_f1(golden_answers, generated_answers)
        contain_result = evaluate_contain(golden_answers, generated_answers)
        
        result = {**em_result, **f1_result}
        result = {**contain_result, **result}

    elif "cola" in test_name.lower():
        result = evaluate_mcc(golden_answers, generated_answers)

    elif "ropes" in test_name.lower():
        
        em_result = evaluate_em(golden_answers, generated_answers)
        f1_overlap_result = evaluate_f1_overlap(golden_answers, generated_answers)
        contain_result = evaluate_contain(golden_answers, generated_answers)
        result = {**em_result, **f1_overlap_result}
        result = {**contain_result, **result}
    
    elif "webq" in test_name.lower() or "race" in test_name.lower() or "freebase" in test_name.lower():
        
        em_result = evaluate_em(golden_answers, generated_answers)
        contain_result = evaluate_contain(golden_answers, generated_answers)
        result = {**em_result, **contain_result}

    elif "sum" in test_name.lower():
        result = evaluate_rouge_scores(golden_answers, generated_answers)
    
    elif "gsm" in test_name.lower():
        golden_answers = [item.split("####")[-1] if "####" in item else item for item in golden_answers]
        generated_answers = [item.split("####")[-1] if "####" in item else item for item in generated_answers]

     
        em_result = evaluate_em(golden_answers, generated_answers)
        contain_result = evaluate_contain(golden_answers, generated_answers)
        result = {**em_result, **contain_result}



    with open(evaluation_datapath, 'w', encoding='utf-8') as json_file:
        path = {"datapath": datapath}
        result = {**path, **result}
        json.dump(result, json_file, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, required=True, help='Path to the JSON data file')
    parser.add_argument('--evaluation_datapath', type=str, required=True, help='Path to save the evaluation results')
    args = parser.parse_args()
    datapath = args.datapath
    evaluation_datapath = args.evaluation_datapath
    evaluation_all(datapath, evaluation_datapath)

    # python metric_evaluation.py --datapath --evaluation_datapath


