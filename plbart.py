import torch
import pickle
import numpy as np
from os.path import join
import logging
import argparse

import evaluate
from datasets import load_dataset
from transformers import PLBartForConditionalGeneration, PLBartTokenizer
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers.models.plbart.modeling_plbart import shift_tokens_right

logger = logging.getLogger(__name__)

bleu_metric = evaluate.load("sacrebleu")
em_metric = evaluate.load("exact_match")

max_length=512
def convert_to_features(example_batch):
    input_encodings = tokenizer.batch_encode_plus(example_batch['src'], 
                                                  pad_to_max_length=True, 
                                                  max_length=max_length,
                                                  truncation=True)
    with tokenizer.as_target_tokenizer():
        target_encodings = tokenizer.batch_encode_plus(example_batch['tgt'], 
                                                       pad_to_max_length=True, 
                                                       max_length=max_length, 
                                                       truncation=True)

    labels = target_encodings['input_ids']
    labels = np.asarray(labels)
    labels[labels[:, :] == model.config.pad_token_id] = -100
    decoder_input_ids = shift_tokens_right(torch.from_numpy(labels), model.config.pad_token_id)

    encodings = {
        'input_ids': input_encodings['input_ids'],
        'attention_mask': input_encodings['attention_mask'],
        'decoder_input_ids': decoder_input_ids.tolist(),
        'labels': labels.tolist(),
    }

    return encodings

def bleu_postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels

def em_postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # Evaluate BLEU
    bleu_preds, bleu_labels = bleu_postprocess_text(decoded_preds, decoded_labels)
    bleu_result = bleu_metric.compute(predictions=bleu_preds, references=bleu_labels)
    # Evaluate EM
    em_preds, em_labels = em_postprocess_text(decoded_preds, decoded_labels)
    em_result = em_result = em_metric.compute(predictions=em_preds, references=em_labels)
    # Combine BLEU and EM
    result = {"sacrebleu": bleu_result["score"], "exact_match": em_result["exact_match"]}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default=None, type=str, required=True)
parser.add_argument("--data_affix", default=None, type=str, required=True)
parser.add_argument("--data_dir", type=str)
parser.add_argument("--model_dir", type=str)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--num_train_epochs", default=25, type=int)
parser.add_argument("--num_beams", default=5, type=int)
args = parser.parse_args()
model_name = args.model_name
data_affix = args.data_affix

#load dataset
data_dir = args.data_dir
data_files = {"train": join(data_dir, "final", "{}_train.json".format(data_affix)),
              "validation": join(data_dir, "final", "{}_val.json".format(data_affix)),
              "test": join(data_dir, "final", "{}_test.json".format(data_affix))}
dataset = load_dataset("json", data_files=data_files, field="data")

#load model and tokenizer
tokenizer = PLBartTokenizer.from_pretrained("uclanlp/{}".format(model_name), src_lang="en_XX", tgt_lang="python")
model = PLBartForConditionalGeneration.from_pretrained("uclanlp/{}".format(model_name))

#transform features
dataset = dataset.map(convert_to_features, batched=True)
columns = ['input_ids', 'labels', 'decoder_input_ids','attention_mask',]
dataset.set_format(type='torch', columns=columns)
dataset = dataset.shuffle(seed=args.seed)

#define training arguments
model_dir = args.model_dir
output_dir = join(model_dir, '{}-{}-{}'.format(model_name, data_affix, args.seed))
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    num_train_epochs=args.num_train_epochs,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    seed=args.seed,
    warmup_ratio=0.01,
    save_strategy="epoch",
    logging_strategy="epoch",
    evaluation_strategy="epoch",
    # do not evaluate for the first few epochs
    eval_delay=30,
    save_total_limit=args.num_train_epochs-30,
    load_best_model_at_end=True,
    metric_for_best_model="sacrebleu",
    greater_is_better=True,
    predict_with_generate=True,
    generation_max_length=max_length,
    generation_num_beams=args.num_beams,
)

#define trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    compute_metrics=compute_metrics,
)

#train
trainer.train()

# evaluate model on the validation set (best model is loaded)
metrics = trainer.evaluate(dataset["validation"], metric_key_prefix="eval")
trainer.log_metrics("eval_best", metrics)
trainer.save_metrics("eval_best", metrics)

# evaluate model on the test set
logger.info("*** Predict and Evaluate on the Test Set ***")

predictions, label_ids, metrics = trainer.predict(dataset["test"], metric_key_prefix="test")
trainer.log_metrics("test", metrics)
trainer.save_metrics("test", metrics)

logger.info("*** Save Model ***")
save_dir = join(output_dir, "best_model")
trainer.save_model(output_dir=save_dir)
trainer.save_state()

logger.info("*** Save predictions ***")
with open(join(save_dir, "preds.pkl"), "wb") as f:
    pickle.dump({"preds": predictions, "label_ids": label_ids}, f)
