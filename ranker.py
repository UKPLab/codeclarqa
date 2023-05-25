import os
from os.path import join
import sys
import random
import pickle
import logging
import wandb

import argparse

import torch
import numpy as np
import pandas as pd

from transformer_rankers.trainers import transformer_trainer
from transformer_rankers.datasets import dataset
from transformer_rankers.negative_samplers import negative_sampling
from transformer_rankers.eval import results_analyses_tools
from transformers import AutoTokenizer, AutoModelForSequenceClassification


wandb.init()

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default=None, type=str, required=True)
parser.add_argument("--data_dir", type=str)
parser.add_argument("--anserini_dir", type=str)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--num_epochs", default=8, type=int)
parser.add_argument("--train_batch_size", default=32, type=int)
parser.add_argument("--eval_batch_size", default=32, type=int)
parser.add_argument("--learning_rate", default=5e-5, type=float)
parser.add_argument("--max_seq_len", default=160, type=int)
parser.add_argument("--negative_sampling_strategy", default="BM25", type=str, required=True)
parser.add_argument("--save_dir", default=None, type=str, required=True)
args = parser.parse_args()


logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s [%(levelname)s] %(message)s",
  handlers=[
      logging.StreamHandler(sys.stdout)
  ]
)

# get the first type + keyword + question
def get_first_type_keyword_question(sample, idx):
    if sample["origin_labels"][idx][0] == 'bipolar':
        return ('bipolar', sample['origin_paths'][idx], sample['origin_QAs'][idx][0])
    elif sample["origin_labels"][idx][0] == 'multiple':
        return ('multiple', sample['origin_labels'][idx][1], sample['origin_QAs'][idx][0])
    else:
        raise NameError('question does not exist')

# load data
data_dir = args.data_dir
dialogue_text_code_data = pickle.load(open(join(data_dir, "final", "all.pkl"), "rb"))
train_dev_test_split = pickle.load(open(join(data_dir, "final", "train_dev_test_split.pkl"), "rb"))

# getting question_bank
try:
    question_bank = pd.read_csv(join(data_dir, "final", "question_bank.tsv"), sep="\t")
except:
    question_bank = []
    for idx in dialogue_text_code_data:
        sample = dialogue_text_code_data[idx]
        assert len(sample["lines"]) == len(sample["origin_paths"]) == len(sample["origin_QAs"]) == len(sample["origin_labels"])
        n_rounds = len(sample["lines"])
        if n_rounds == 0:
            continue

        for i in range(n_rounds):
            sample_tuple = get_first_type_keyword_question(sample, i)
            if sample_tuple not in question_bank:
                question_bank.append(sample_tuple)

    question_bank = sorted(question_bank, key=lambda tup: tup[0], reverse=True)
    question_bank = pd.DataFrame(question_bank, columns =['type', 'keyword', 'question'])
    question_bank.to_csv(join(data_dir, "final", "question_bank.tsv"), sep="\t", index=False)

logging.info("Finish generating the question bank.")

# getting dataset
def gen_clarQ_dataset(full_dataset, split, split_type):
    split_set = []
    for idx in split:
        sample = full_dataset[idx]
        n_rounds = len(sample["lines"])
        if n_rounds == 0:
            continue
        for i in range(n_rounds):
            sample_tuple = get_first_type_keyword_question(sample, i)
            split_set.append((sample['text'], sample_tuple[0], sample_tuple[1], sample_tuple[2]))
    split_set = pd.DataFrame(split_set, columns =['text', 'type', 'keyword', 'question'])
    split_set.to_csv(join(data_dir, "final", "clarQ_{}.tsv".format(split_type)), sep="\t", index=False)
    return split_set

# implement the defined function
train_split, dev_split, test_split = train_dev_test_split
try:
    train = pd.read_csv(join(data_dir, "final", "clarQ_train.tsv"), sep="\t")
    dev = pd.read_csv(join(data_dir, "final", "clarQ_dev.tsv"), sep="\t")
    test = pd.read_csv(join(data_dir, "final", "clarQ_test.tsv"), sep="\t")
except:
    train = gen_clarQ_dataset(dialogue_text_code_data, train_split, "train")
    dev = gen_clarQ_dataset(dialogue_text_code_data, dev_split, "dev")
    test = gen_clarQ_dataset(dialogue_text_code_data, test_split, "test")

train = train[['text', 'question']]
dev = dev[['text', 'question']]
test = test[['text', 'question']]

logging.info("Finish generating the clarQ dataset.")

# define the model
seed = args.seed
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

#Lets use an almost balanced amount of positive and negative samples during training.
average_relevant_per_query = train.groupby("text").count().mean().values[0]

#Instantiate BM25 negative sampler.
if args.negative_sampling_strategy == "BM25":
    # get the BM25 negative sampler
    ns_train = negative_sampling.BM25NegativeSamplerPyserini(list(question_bank["question"].values), int(average_relevant_per_query), join(data_dir, "final", "anserini_train"), -1, args.anserini_dir)
    ns_val = negative_sampling.BM25NegativeSamplerPyserini(list(question_bank["question"].values), int(average_relevant_per_query), join(data_dir, "final", "anserini_train"), -1, args.anserini_dir)

elif args.negative_sampling_strategy == "Random":
    # get the random negative sampler, which doesn't need anserini
    ns_train = negative_sampling.RandomNegativeSampler(list(question_bank["question"].values), int(average_relevant_per_query))
    ns_val = negative_sampling.RandomNegativeSampler(list(question_bank["question"].values), int(average_relevant_per_query))

#Create the loaders for the dataset, with the respective negative samplers
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
dataloader = dataset.QueryDocumentDataLoader(train_df=train,
                    val_df=dev, test_df=dev,
                    tokenizer=tokenizer, negative_sampler_train=ns_train,
                    negative_sampler_val=ns_val, task_type='classification',
                    train_batch_size=args.train_batch_size , val_batch_size=args.eval_batch_size , max_seq_len=args.max_seq_len,
                    sample_data=-1, cache_path=join(data_dir, "final", "cache"))

train_loader, val_loader, test_loader = dataloader.get_pytorch_dataloaders()

#Use BERT (any model that has SequenceClassification class from HuggingFace would work here)
model = AutoModelForSequenceClassification.from_pretrained(args.model_name)

#Instantiate trainer that handles fitting.
trainer = transformer_trainer.TransformerTrainer(model=model,
  train_loader=train_loader,
  val_loader=val_loader, test_loader=test_loader, num_ns_eval=30, # num_ns_eval=int(average_relevant_per_query)
  task_type="classification", tokenizer=tokenizer,
  validate_every_epochs=1, num_validation_batches=-1, validation_metric="recip_rank",
  num_epochs=args.num_epochs, lr=args.learning_rate, sacred_ex=None, saving_dir=join(args.save_dir+"_"+str(args.seed), args.model_name, args.negative_sampling_strategy))

#Train (our validation eval uses the NS sampling procedure)
trainer.fit()