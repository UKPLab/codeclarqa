import json
from os.path import join
import numpy as np
import pandas as pd

import pickle
import argparse

from datasets import load_dataset

def load_json_file(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return data

def load_pickle_file(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data

def get_first_type_keyword_question(sample, idx):
    if sample["origin_labels"][idx][0] == 'bipolar':
        return 'bipolar' + ':' + sample['origin_paths'][idx]
    elif sample["origin_labels"][idx][0] == 'multiple':
        return 'multiple' + ':' + sample['origin_labels'][idx][1]
    else:
        raise NameError('question does not exist')

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--model_dir", type=str)
parser.add_argument("--clariq_repo_dir", type=str)
parser.add_argument("--best_model_classifier", default="bert-base-cased", type=str)
parser.add_argument("--best_model_ranker", default="facebook--bart-base_Random_epoch10", type=str)
args = parser.parse_args()
data_dir = args.data_dir

# get the top 30 documents from each query
run_df = pd.read_csv('{}/runs_{}/{}'.format(args.clariq_repo_dir, args.seed, args.best_model_ranker), 
                    sep=' ', 
                    header=None)
run_df = run_df.sort_values(by=[0, 4], ascending=False).drop_duplicates(subset=[0, 4], keep='first')
run_question_set_list = run_df.groupby(0)[2].agg(list).to_dict()

# get the indices for each query
ranker_pred = pd.read_csv(open(join(data_dir, "final", "clarQ_pred_with_text_index.tsv")), sep='\t')
ranker_indices = ranker_pred['text_index'].tolist()

# get the question bank
question_bank_path = join(data_dir, "final", "question_bank_with_question_id.tsv")
question_bank = pd.read_csv(question_bank_path, sep='\t')

# get the predictions from classifier
classifier_preds = load_pickle_file("{}/{}_{}/best_model/preds.pkl".format(args.model_dir, args.best_model_classifier, args.seed))
classifier_preds = classifier_preds['preds']
classifier_preds = np.argmax(classifier_preds, axis=1).tolist()

# get dialogue text code data
dialogue_text_code_data = pickle.load(open(join(data_dir, "final", "all.pkl"), "rb"))
train_dev_test_split = pickle.load(open(join(data_dir, "final", "train_dev_test_split.pkl"), "rb"))
test_split = train_dev_test_split[2]

# create dataset with answered QAs
top_1 = {}
top_3 = {}
top_5 = {}
for i, idx in enumerate(test_split):
    # get the sample
    sample = dialogue_text_code_data[idx]
    # get the top 30 documents
    ranker_idx = ranker_indices[i]
    most_freq_questions = run_question_set_list[ranker_idx]
    # get the top 1, 3, 5 questions
    top_1[idx] = []
    top_3[idx] = []
    top_5[idx] = []
    sample_text_index = []
    num_lines = len(sample['lines'])
    if num_lines == 0:
        continue
    if classifier_preds[i] == 0:
        continue
    for j in range(num_lines):
        step_qid = get_first_type_keyword_question(sample, j)
        assert step_qid in question_bank['question_id'].tolist()
        if step_qid in most_freq_questions[:1]:
            top_1[idx].append(j)
        if step_qid in most_freq_questions[:3]:
            top_3[idx].append(j)
        if step_qid in most_freq_questions[:5]:
            top_5[idx].append(j)

# build code generation dataset
def get_code_gen_data_json(all_data, ids, top_k):
    output_data = {"data": []}
    for idx in ids:
        str_idx = str(idx)
        basic_temp = all_data[str_idx]["text"]
        sample_QAs = all_data[str_idx]["origin_QAs"]
        len_QAs = len(sample_QAs)
        for qid in top_k[idx]:
            basic_temp += "</s>{}</s>{}".format(sample_QAs[qid][0],
                                                sample_QAs[qid][1])

        output_data["data"].append({"src": basic_temp,
                                    "tgt": all_data[str_idx]["code"]})

    return output_data

# create dataset without answered QAs
top_1_wo = {}
top_3_wo = {}
top_5_wo = {}
for i, idx in enumerate(test_split):
    # get the sample
    sample = dialogue_text_code_data[idx]
    # get the top 30 documents
    ranker_idx = ranker_indices[i]
    most_freq_questions = run_question_set_list[ranker_idx]
    top_1_wo[idx] = []
    top_3_wo[idx] = []
    top_5_wo[idx] = []
    if classifier_preds[i] == 0:
        continue
    assert most_freq_questions[0] in question_bank['question_id'].tolist()
    sample_freq_questions = question_bank[question_bank['question_id']==most_freq_questions[0]]['question'].tolist()
    assert len(sample_freq_questions) == 1
    sample_freq_question = sample_freq_questions[0]
    top_1_wo[idx].append(sample_freq_question)

    for elem in most_freq_questions[:3]:
        assert elem in question_bank['question_id'].tolist()
        sample_freq_questions = question_bank[question_bank['question_id']==elem]['question'].tolist()
        assert len(sample_freq_questions) == 1
        sample_freq_question = sample_freq_questions[0]
        top_3_wo[idx].append(sample_freq_question)
    
    for elem in most_freq_questions[:5]:
        assert elem in question_bank['question_id'].tolist()
        sample_freq_questions = question_bank[question_bank['question_id']==elem]['question'].tolist()
        assert len(sample_freq_questions) == 1
        sample_freq_question = sample_freq_questions[0]
        top_5_wo[idx].append(sample_freq_question)


# build code generation dataset
def get_code_gen_data_json_wo(all_data, ids, top_k_wo):
    output_data = {"data": []}
    for idx in ids:
        str_idx = str(idx)
        basic_temp = all_data[str_idx]["text"]
        sample_QAs = all_data[str_idx]["origin_QAs"]
        len_QAs = len(sample_QAs)
        for elem in top_k_wo[idx]:
            basic_temp += "</s>{}".format(elem)

        output_data["data"].append({"src": basic_temp,
                                    "tgt": all_data[str_idx]["code"]})

    return output_data

top_1_test = get_code_gen_data_json(dialogue_text_code_data, test_split, top_1)
top_3_test = get_code_gen_data_json(dialogue_text_code_data, test_split, top_3)
top_5_test = get_code_gen_data_json(dialogue_text_code_data, test_split, top_5)


with open(join(data_dir, 'final', 'gen_code_test_top_1_{}.json'.format(args.seed)), 'w') as outfile:
    outfile.write(json.dumps(top_1_test))
with open(join(data_dir, 'final', 'gen_code_test_top_3_{}.json'.format(args.seed)), 'w') as outfile:
    outfile.write(json.dumps(top_3_test))
with open(join(data_dir, 'final', 'gen_code_test_top_5_{}.json'.format(args.seed)), 'w') as outfile:
    outfile.write(json.dumps(top_5_test))

top_1_test_wo = get_code_gen_data_json_wo(dialogue_text_code_data, test_split, top_1_wo)
top_3_test_wo = get_code_gen_data_json_wo(dialogue_text_code_data, test_split, top_3_wo)
top_5_test_wo = get_code_gen_data_json_wo(dialogue_text_code_data, test_split, top_5_wo)

with open(join(data_dir, 'final', 'gen_code_test_top_1_wo_{}.json'.format(args.seed)), 'w') as outfile:
    outfile.write(json.dumps(top_1_test_wo))
with open(join(data_dir, 'final', 'gen_code_test_top_3_wo_{}.json'.format(args.seed)), 'w') as outfile:
    outfile.write(json.dumps(top_3_test_wo))
with open(join(data_dir, 'final', 'gen_code_test_top_5_wo_{}.json'.format(args.seed)), 'w') as outfile:
    outfile.write(json.dumps(top_5_test_wo))