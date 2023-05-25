import torch
import numpy as np
from os.path import join

import argparse

import torch
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import DataCollatorForLanguageModeling

def tokenize(element):
    str_inputs = []
    for src, tgt in zip(element["src"], element["tgt"]):
        str_inputs.append("# {} <|python|> \n{}".format(src.replace("</s>", " "), tgt))
    print(len(str_inputs))
    print(str_inputs[0], end="\n\n")
    inputs = tokenizer(str_inputs, truncation=True, max_length=max_length, return_overflowing_tokens=True, return_length=True)
    input_batch = []
    for length, input_ids in zip(inputs["length"], inputs["input_ids"]):
        if length == max_length:
            input_batch.append(input_ids)
    return {"input_ids": input_batch}

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default=None, type=str, required=True)
parser.add_argument("--data_affix", default=None, type=str, required=True)
parser.add_argument("--data_dir", type=str)
parser.add_argument("--model_dir", type=str)
parser.add_argument("--max_length", default=128, type=int)
parser.add_argument("--seed", default=42, type=int)
args = parser.parse_args()
model_name = args.model_name
data_affix = args.data_affix
max_length = args.max_length

#load dataset
data_dir = args.data_dir
data_files = {"train": join(data_dir, "final", "{}_train.json".format(data_affix)),
              "validation": join(data_dir, "final", "{}_val.json".format(data_affix)),
              "test": join(data_dir, "final", "{}_test.json".format(data_affix))}
dataset = load_dataset("json", data_files=data_files, field="data")


#load model and tokenizer
# EleutherAI/gpt-neo-125M AutoModelForCausalLM
# codeparrot/codeparrot-small AutoModelForCausalLM
# codeparrot/codeparrot-small-text-to-code AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

#transform features
print(dataset["train"].column_names)
tokenized_datasets = dataset.map(tokenize, batched=True, remove_columns=dataset["train"].column_names)

# Data collator
# Note that some special tokens in GPT2 tokenizer do not exist
# SPECIAL_TOKENS  = { "bos_token": "<|BOS|>",
#                    "eos_token": "<|EOS|>",
#                     "unk_token": "<|UNK|>",                    
#                     "pad_token": "<|PAD|>",
#                     "sep_token": "<|SEP|>"}
# tokenizer.add_special_tokens(SPECIAL_TOKENS)
tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

#define training arguments
model_dir = args.model_dir
training_args = TrainingArguments(
    output_dir=join(model_dir, '{}-{}-{}'.format(model_name.split("/")[-1], data_affix, args.seed)),
    num_train_epochs=40,
    learning_rate=5e-5,
    save_total_limit=5,
    per_device_train_batch_size=32,
    gradient_accumulation_steps=1,
    seed=args.seed,
    warmup_ratio=0.01,
    save_strategy="epoch",
    logging_strategy="epoch",
)

#define trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation']
)

#train
trainer.train()