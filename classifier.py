from os.path import join

import json
import pickle
import numpy as np
from os.path import join
import logging
import argparse

import torch
import evaluate
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding

logger = logging.getLogger(__name__)

# build classification dataset
def get_cls_data_json(all_data, ids):
    output_data = {"data": []}
    for idx in ids:
        str_idx = str(idx)
        basic_temp = all_data[str_idx]["text"]
        sample_QAs = all_data[str_idx]["origin_QAs"]
        len_QAs = len(sample_QAs)
        output_data["data"].append({"sentence1": basic_temp,
                                    "label": int(len_QAs != 0),
                                    "idx": str_idx})

    return output_data

def tokenize_function(example):
    return tokenizer(example["sentence1"], truncation=True, padding="max_length", max_length=128)

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default=None, type=str, required=True)
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--model_dir", type=str, required=True)
parser.add_argument("--train_batch_size", default=32, type=int)
parser.add_argument("--eval_batch_size", default=32, type=int)
parser.add_argument("--num_train_epochs", default=10, type=int)
parser.add_argument("--weight_decay", default=0.01, type=float)
parser.add_argument("--learning_rate", default=5e-5, type=float)
parser.add_argument("--seed", default=42, type=int)
args = parser.parse_args()

#load dataset
data_dir = args.data_dir
dialogue_text_code_data = pickle.load(open(join(data_dir, "final", "all.pkl"), "rb"))
train_dev_test_split = pickle.load(open(join(data_dir, "final", "train_dev_test_split.pkl"), "rb"))
train_split = train_dev_test_split[0]
dev_split = train_dev_test_split[1]
test_split = train_dev_test_split[2]

data_files = {"train": join(data_dir, "final", "cls_train.json"),
              "validation": join(data_dir, "final", "cls_val.json"),
              "test": join(data_dir, "final", "cls_test.json")}

try:
    dataset = load_dataset("json", data_files=data_files, field="data")
except:
    train_data = get_cls_data_json(dialogue_text_code_data, train_split)
    val_data = get_cls_data_json(dialogue_text_code_data, dev_split)
    test_data = get_cls_data_json(dialogue_text_code_data, test_split)
    with open(join(data_dir, 'final', 'cls_train.json'), 'w') as outfile:
        outfile.write(json.dumps(train_data))
    with open(join(data_dir, 'final', 'cls_val.json'), 'w') as outfile:
        outfile.write(json.dumps(val_data))
    with open(join(data_dir, 'final', 'cls_test.json'), 'w') as outfile:
        outfile.write(json.dumps(test_data))

    dataset = load_dataset("json", data_files=data_files, field="data")

# load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

# get tokenized dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# get data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# define compute metrics
def compute_metrics(eval_pred):    
    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    predictions = np.argmax(predictions, axis=1)
    acc = acc_metric.compute(predictions=predictions, references=labels)
    precision = precision_metric.compute(predictions=predictions, references=labels)
    recall = recall_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels)
    return {**acc, **precision, **recall, **f1}

# define evaluation metric
acc_metric = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")

# get training arguments
model_dir = args.model_dir
training_args = TrainingArguments(
    output_dir=join(model_dir, "{}_{}".format(args.model_name.split("/")[-1], args.seed)),
    do_train=True, do_eval=True, do_predict=True,
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="epoch",
    learning_rate=args.learning_rate,
    per_device_train_batch_size=args.train_batch_size,
    per_device_eval_batch_size=args.eval_batch_size,
    num_train_epochs=args.num_train_epochs,
    weight_decay=args.weight_decay,
    logging_dir=join(model_dir, "{}_{}".format(args.model_name.split("/")[-1], args.seed), "logs"),
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    seed=args.seed,
)

# get trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# train model
trainer.train()

# evaluate model on the validation set (best model is loaded)
metrics = trainer.evaluate(tokenized_dataset["validation"], metric_key_prefix="eval")
trainer.log_metrics("eval_best", metrics)
trainer.save_metrics("eval_best", metrics)

# evaluate model on the test set
logger.info("*** Predict and Evaluate on the Test Set ***")

predictions, label_ids, metrics = trainer.predict(tokenized_dataset["test"], metric_key_prefix="test")
trainer.log_metrics("test", metrics)
trainer.save_metrics("test", metrics)

logger.info("*** Save Model ***")
save_dir = join(model_dir, "{}_{}".format(args.model_name.split("/")[-1], args.seed), "best_model")
trainer.save_model(output_dir=save_dir)
trainer.save_state()

logger.info("*** Save predictions ***")
with open(join(save_dir, "preds.pkl"), "wb") as f:
    pickle.dump({"preds": predictions, "label_ids": label_ids}, f)