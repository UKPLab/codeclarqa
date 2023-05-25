import torch
import numpy as np
from os.path import join
from tqdm import tqdm
import json
from os import listdir

import logging

import argparse

import evaluate
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data.dataloader import DataLoader

logger = logging.getLogger(__name__)

#metric
bleu_metric = evaluate.load("sacrebleu")
em_metric = evaluate.load("exact_match")

def bleu_postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels

def em_postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    return preds, labels

def compute_metrics(decoded_preds, decoded_labels):
    # Evaluate BLEU
    bleu_preds, bleu_labels = bleu_postprocess_text(decoded_preds, decoded_labels)
    bleu_result = bleu_metric.compute(predictions=bleu_preds, references=bleu_labels)
    # Evaluate EM
    em_preds, em_labels = em_postprocess_text(decoded_preds, decoded_labels)
    em_result = em_metric.compute(predictions=em_preds, references=em_labels)
    # Combine BLEU and EM
    result = {"sacrebleu": bleu_result["score"], "exact_match": em_result["exact_match"]}
    prediction_lens = [len(tokenizer.encode(pred)) for pred in decoded_preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

def model_evaluate(args, evaluate_dataset, model):
    model.eval()
    all_preds = []
    all_labels = []
    device = torch.device("cuda:0")
    model = model.to(device)
    len_dataset = len(evaluate_dataset)
    for i in tqdm(range(len_dataset)):
        sample = evaluate_dataset[i]
        src = sample["src"]
        # transform the src to the format of GPT2 input
        src = "# {} <|python|> \n".format(src.replace("</s>", " "))
        tgt = sample["tgt"]
        input_ids = tokenizer.encode(src)
        sent_len = len(input_ids)
        input_ids = torch.tensor(input_ids).unsqueeze(0)
        input_ids = input_ids.to(device)
        # generate the predictions
        preds  = model.generate(input_ids, 
                                do_sample=True, 
                                max_length=sent_len+args.max_length,
                                num_beams=5,
                                num_return_sequences=1,
                                early_stopping=True)
        # check if the prediction starts with the correct token
        assert preds [:, :input_ids.shape[-1]].tolist() == input_ids.tolist()
        assert preds.shape[0] == 1
        preds = preds[:, input_ids.shape[-1]:]
        preds = preds.tolist()[0]
        # decode the tokens and add to the list
        preds = tokenizer.decode(preds)
        all_preds.append(preds)
        all_labels.append(tgt)
        # make sure the lengths are correct
        assert len(all_preds) == len(all_labels)
    
    result = compute_metrics(all_preds, all_labels)

    return result

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default=None, type=str, required=True)
parser.add_argument("--data_affix", default=None, type=str, required=True)
parser.add_argument("--data_dir", type=str)
parser.add_argument("--model_dir", type=str)
parser.add_argument("--max_length", default=128, type=int)
parser.add_argument("--num_beams", default=5, type=int, required=False)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--eval_delay", default=0, type=int)
args = parser.parse_args()


#load dataset
data_dir = args.data_dir
data_files = {"train": join(data_dir, "final", "{}_train.json".format(args.data_affix)),
              "validation": join(data_dir, "final", "{}_val.json".format(args.data_affix)),
              "test": join(data_dir, "final", "{}_test.json".format(args.data_affix))}
dataset = load_dataset("json", data_files=data_files, field="data")
val_dataset = dataset["validation"]
test_dataset = dataset["test"]


#tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model_name)

# load model and evaluation
# EleutherAI/gpt-neo-125M
# codeparrot/codeparrot-small
# codeparrot/codeparrot-small-text-to-code
model_dir = args.model_dir
final_model_dir = join(model_dir,'{}-{}-{}'.format(args.model_name.split('/')[-1], args.data_affix, args.seed))
all_checkpoints = listdir(final_model_dir)
all_checkpoints = [e for e in all_checkpoints if e.startswith("checkpoint")]
all_steps = sorted([int(e.split("-")[1]) for e in all_checkpoints])
n_epochs = len(all_steps)

logger.info("n_epochs: {}".format(n_epochs))

all_results = {"best_epoch": -1,
               "best_step": -1,
               "best_eval_bleu": 0,
               "best_eval_metric": {},
               "test_metric": {},
               "log_history":{}}

for i, step in enumerate(all_steps):
    # print epoch
    epoch = i+1+args.eval_delay
    logger.info("epoch: {}".format(epoch))

    # check eval delay
    if epoch <= args.eval_delay:
        continue

    # load model
    model = AutoModelForCausalLM.from_pretrained(join(final_model_dir,"checkpoint-{}".format(step)))

    # evaluation
    results = model_evaluate(args, val_dataset, model)

    # save the results
    all_results["log_history"][epoch] = results

    # print the results
    logger.info("validation results: \n{}".format(results))

    # update best evaluation bleu
    if results["sacrebleu"] > all_results["best_eval_bleu"]:
        all_results["best_eval_bleu"] = results["sacrebleu"]
        all_results["best_eval_metric"] = results
        all_results["best_epoch"] = epoch
        all_results["best_step"] = step
        all_results["best_eval_metric"] = results

best_checkpoint_dir = join(final_model_dir,"checkpoint-{}".format(all_results["best_step"]))
model = AutoModelForCausalLM.from_pretrained(best_checkpoint_dir)

# test on dataset
test_results = model_evaluate(args, test_dataset, model)
# print the test results
logger.info("test results: \n{}".format(test_results))

# update test results on the best model
all_results["test_metric"] = test_results

# update all results
with open(join(final_model_dir, "all_results.json"), "w") as f:
    json.dump(all_results, f)