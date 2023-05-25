import os
from os import listdir
from os.path import join
import sys
import random
import pickle
import logging
import argparse
import wandb
from tqdm import tqdm

import torch

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
import nltk
from nltk.stem.porter import PorterStemmer

from transformers.data.data_collator import default_data_collator
from transformers.data.processors.utils import InputFeatures
from torch.utils.data import Dataset, DataLoader

from transformer_rankers.utils import utils

from transformer_rankers.trainers import transformer_trainer
from transformer_rankers.datasets import dataset
from transformer_rankers.negative_samplers import negative_sampling
from transformer_rankers.eval import results_analyses_tools
from transformers import AutoTokenizer, AutoModelForSequenceClassification

wandb.init()

# stem toknizer
def stem_tokenize(text, remove_stopwords=True):
  stemmer = PorterStemmer()
  tokens = [word for sent in nltk.sent_tokenize(text) \
                                      for word in nltk.word_tokenize(sent)]
  tokens = [word for word in tokens if word not in \
          nltk.corpus.stopwords.words('english')]
  return [stemmer.stem(word) for word in tokens]


# getting dataset
def gen_clarQ_dataset(full_dataset, split, split_type):
    split_set = []
    for idx in split:
        sample = full_dataset[idx]
        split_set.append((sample['text'], 
                          'multiple', 
                          'array', 
                          "Do you want to call anything related to \'array\'? If yes, which one?"))
    split_set = pd.DataFrame(split_set, columns =['text', 'type', 'keyword', 'question'])
    split_set.to_csv(join(data_dir, "final", "clarQ_{}.tsv".format(split_type)), sep="\t", index=False)
    return split_set

# parse args
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default=None, type=str, required=True)
parser.add_argument("--data_dir", type=str)
parser.add_argument("--code_dir", type=str)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--num_epochs", default=10, type=int)
parser.add_argument("--train_batch_size", default=32, type=int)
parser.add_argument("--eval_batch_size", default=32, type=int)
parser.add_argument("--learning_rate", default=5e-5, type=float)
parser.add_argument("--max_seq_len", default=160, type=int)
parser.add_argument("--negative_sampling_strategy", default="Random", type=str, required=True)
parser.add_argument("--save_dir", default=None, type=str, required=True)
# directory to the predictions file, later used for evaluation by the official script from ClariQ-repo
parser.add_argument("--clariq_repo_dir", default=None, type=str, required=True)
args = parser.parse_args()

# Files paths
data_dir = args.data_dir
request_file_path = join(data_dir, "final", "clarQ_pred.tsv")
question_bank_path = join(data_dir, "final", "question_bank.tsv")
run_file_path = join(args.clariq_repo_dir ,'runs_{}'.format(args.seed))

# load data
dialogue_text_code_data = pickle.load(open(join(data_dir, "final", "all.pkl"), "rb"))
train_dev_test_split = pickle.load(open(join(data_dir, "final", "train_dev_test_split.pkl"), "rb"))

# implement the defined function
train_split, dev_split, test_split = train_dev_test_split
try:
    pred = pd.read_csv(join(data_dir, "final", "clarQ_pred.tsv"), sep="\t")
except:
    pred = gen_clarQ_dataset(dialogue_text_code_data, test_split, "pred")

# Reads files and build bm25 corpus (index)
# try to read clarQ_pred_with_text_index.tsv
try:
    pred = pd.read_csv(join(data_dir, "final", "clarQ_pred_with_text_index.tsv"), sep='\t')
except:
    pred = pd.read_csv(request_file_path, sep='\t')
    unique_text = pred['text'].unique()
    unique_text_index = {text: i for i, text in enumerate(unique_text)}
    pred["text_index"] = pred["text"].apply(lambda x: unique_text_index[x])
    pred['question_id'] = pred['type'] + ':' + pred['keyword']
    # save pred file
    pred.to_csv(join(args.data_dir, "final", "clarQ_pred_with_text_index.tsv"), sep='\t', index=False)

# question bank
question_bank = pd.read_csv(join(args.data_dir, "final", "question_bank_with_question_id.tsv"), sep='\t')

# full retrieval
class SimpleDataset(Dataset):
    def __init__(self, features):
        self.features = features
    def __len__(self):
        return len(self.features)
    def __getitem__(self, index):
        return self.features[index]

#Lets not use the null document for no question.
all_documents = list(question_bank["question"].values)
examples = []
for tid in pred['text_index'].unique():
    query = pred.loc[pred['text_index']==tid, 'text'].tolist()[0]
    for doc in all_documents:
      examples.append((query, doc))

# Convert to features
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
batch_encoding = tokenizer.batch_encode_plus(examples, max_length=args.max_seq_len, pad_to_max_length=True)

features = []
for i in range(len(examples)):
    inputs = {k: batch_encoding[k][i] for k in batch_encoding}
    feature = InputFeatures(**inputs, label=0)
    features.append(feature)

# Create dataloader
dataset = SimpleDataset(features)
dataloader = DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=default_data_collator)

model = AutoModelForSequenceClassification.from_pretrained(join(args.save_dir, 
                                                                args.model_name, 
                                                                args.negative_sampling_strategy, 
                                                                "epoch_{}".format(args.num_epochs)))
device = torch.device("cuda:0")
model.to(device)
trainer = transformer_trainer.TransformerTrainer(model=model, train_loader=None, val_loader=None, test_loader=None, # num_ns_eval=int(average_relevant_per_query)
                                                num_ns_eval=30, task_type="classification", tokenizer=tokenizer, validate_every_epochs=1, 
                                                num_validation_batches=-1, validation_metric="recip_rank", num_epochs=args.num_epochs, 
                                                lr=args.learning_rate, sacred_ex=None, saving_dir=join(args.save_dir, args.model_name, args.negative_sampling_strategy))

# Run model
logits, _, softmax_output = trainer.predict(dataloader)
softmax_output_by_query = utils.acumulate_list(softmax_output[0], len(all_documents))

# full reranking
model_run_file_path = join(run_file_path, "pred_{}_{}_epoch{}".format(args.model_name.replace("/", "--"), 
                                                                     args.negative_sampling_strategy, 
                                                                     args.num_epochs))
all_doc_ids = np.array(question_bank["question_id"].values)
with open(model_run_file_path, 'w') as fo:
    for tid_idx, tid in enumerate(pred['text_index'].unique()):
        all_documents_scores = np.array(softmax_output_by_query[tid_idx])
        top_30_scores_idx = (-all_documents_scores).argsort()[:30]  
        preds = all_doc_ids[top_30_scores_idx]
        for i, qid in enumerate(preds):    
            fo.write('{} 0 {} {} {} BERT-ranker\n'.format(tid, qid, i, len(preds)-i))