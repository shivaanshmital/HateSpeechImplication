import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM

dataset_train = pd.read_csv('train_dataset_final.csv')
dataset_test = pd.read_csv('test_dataset_final.csv')
dataset_val = pd.read_csv('val_dataset_final.csv')
print(dataset_test.columns)
prompt = """\n\nWhat is the implication of this post."""

dataset_train = dataset_train[dataset_train['offensiveYN'] == 1]
dataset_test = dataset_test[dataset_test['offensiveYN'] == 1]
dataset_val = dataset_val[dataset_val['offensiveYN'] == 1]

train_dataset = dataset_train[['post', 'targetStereotype']]
test_dataset = dataset_test[['post', 'targetStereotype']]
val_dataset = dataset_val[['post', 'targetStereotype']]
print(train_dataset.columns)
print(len(train_dataset))
print(len(test_dataset))
print(len(val_dataset))
dataset_train = train_dataset
dataset_test = test_dataset
dataset_val = val_dataset

dataset_train['input'] = None
dataset_train['output'] = None

dataset_val['input'] = None
dataset_val['output'] = None

dataset_test['input'] = None
dataset_test['output'] = None
post_prompt2 = "Implication is:\n"


dataset_train.reset_index(drop=True, inplace=True)
dataset_val.reset_index(drop=True, inplace=True)
dataset_test.reset_index(drop=True, inplace=True)
# print(dataset_train)

def list2samples(example, dataset):
    # print(example)
    dataset.loc[example, 'input'] = dataset.loc[example, 'post'] + prompt
    dataset.loc[example, 'output'] = post_prompt2 + dataset.loc[example, 'targetStereotype']


for i in range(len(dataset_train)):
    # print(dataset_train.loc[i])
    list2samples(i, dataset_train)

for i in range(len(dataset_val)):
    # print(dataset_val.loc[i])
    list2samples(i, dataset_val)
    
for i in range(len(dataset_test)):
    # print(dataset_test.loc[i])
    list2samples(i, dataset_test)
    
print(dataset_train.loc[0])
print(dataset_test.loc[0])
print(dataset_val.loc[0])


import torch
import torch
# print(torch.cuda.is_initialized())
import numpy as np
# print(torch.cuda.is_initialized())
import datasets
# print(torch.cuda.is_initialized())

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)
# print(torch.cuda.is_initialized())

from tabulate import tabulate
# print(torch.cuda.is_initialized())
import nltk
# print(torch.cuda.is_initialized())
from datetime import datetime
# print(torch.cuda.is_initialized())
# device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
# print(device)
# device = "cpu"

device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
print(device)

train_x_json = {
    "input": [],
    "output": []
}
val_x_json = {
    "input": [],
    "output": []
}

test_x_json = {
    "input": [],
    "output": []
}

for i in range(len(dataset_train)):
    train_x_json['input'].append(dataset_train.loc[i, "input"])
    train_x_json['output'].append(dataset_train.loc[i, "output"])
    
for i in range(len(dataset_val)):
    val_x_json['input'].append(dataset_val.loc[i, "input"])
    val_x_json['output'].append(dataset_val.loc[i, "output"])
    
for i in range(len(dataset_test)):
    test_x_json['input'].append(dataset_test.loc[i, "input"])
    test_x_json['output'].append(dataset_test.loc[i, "output"])
    
from datasets import Dataset
train_dataset = Dataset.from_dict(train_x_json)
val_dataset = Dataset.from_dict(val_x_json)
test_dataset = Dataset.from_dict(test_x_json)

language = "english"
# from accelerate import Accelerator
# accelerator = Accelerator(mixed_precision="fp16",gradient_accumulation_steps=1)
model_name = "facebook/bart-large-cnn"
save_name = "bart-large-cnn"
# model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
# model = accelerator.prepare(model)
tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer = accelerator.prepare(tokenizer)
# Set model parameters or use the default
# print(model.config)
# tokenization


encoder_max_length = 512
decoder_max_length = 256

def batch_tokenize_preprocess(batch, tokenizer, max_source_length, max_target_length):
    source, target = batch["input"], batch["output"]
    source_tokenized = tokenizer(
        source, padding="max_length", truncation=True, max_length=max_source_length
    )
    target_tokenized = tokenizer(
        target, padding="max_length", truncation=True, max_length=max_target_length
    )

    batch = {k: v for k, v in source_tokenized.items()}
    # Ignore padding in the loss
    batch["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in l]
        for l in target_tokenized["input_ids"]
    ]
    return batch


train_data = train_dataset.map(
    lambda batch: batch_tokenize_preprocess(
        batch, tokenizer, encoder_max_length, decoder_max_length
    ),
    batched=True,
    remove_columns=train_dataset.column_names,
)

validation_data = val_dataset.map(
    lambda batch: batch_tokenize_preprocess(
        batch, tokenizer, encoder_max_length, decoder_max_length
    ),
    batched=True,
    remove_columns=val_dataset.column_names,
)

test_data = test_dataset.map(
    lambda batch: batch_tokenize_preprocess(
        batch, tokenizer, encoder_max_length, decoder_max_length
    ),
    batched=True,
    remove_columns=test_dataset.column_names,
)

test_data.set_format(type="torch")
train_data.set_format(type="torch")
validation_data.set_format(type="torch")

from torch.utils.data import DataLoader

def create_dataloaders(train_batch_size=8, eval_batch_size=8):
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=train_batch_size)
    val_dataloader = DataLoader(validation_data, shuffle=False, batch_size=eval_batch_size)
    # test_dataloader= DataLoader(test_data, shuffle=False, batch_size=eval_batch_size)
    return train_dataloader, val_dataloader

hyperparameters = {
    "learning_rate": 0.0001,
    "num_epochs": 1000, # set to very high number
    "train_batch_size": 1, # Actual batch size will this x 8 (was 8 before but can cause OOM)
    "eval_batch_size": 1, # Actual batch size will this x 8 (was 32 before but can cause OOM)
    "seed": 42,
    "patience": 3, # early stopping
    "output_dir": "./Models/" + save_name + "/"
}

# Borrowed from https://github.com/huggingface/transformers/blob/master/examples/seq2seq/run_summarization.py
import nltk
nltk.download("punkt", quiet=True)

metric = datasets.load_metric("rouge")


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    # Extract a few results from ROUGE
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
    ]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

import torch
import numpy as np
import datasets

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer
)

from tabulate import tabulate
import nltk
from datetime import datetime
device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"

# model_name = "lucadiliello/bart-small"
# save_name = "bartsmallmodel"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
# model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
# model = accelerator.prepare(model)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Borrowed from https://github.com/huggingface/transformers/blob/master/examples/seq2seq/run_summarization.py

nltk.download("punkt", quiet=True)

metric = datasets.load_metric("rouge")


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    # Extract a few results from ROUGE
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
    ]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

from transformers import DataCollatorForLanguageModeling

training_args = Seq2SeqTrainingArguments(
    output_dir="results",
    num_train_epochs=15,  # demo
    do_train=True,
    do_eval=True,
    per_device_train_batch_size=8,  # demo
    per_device_eval_batch_size=4,
    learning_rate=3e-05,
    warmup_steps=400,
    weight_decay=0.1,
    label_smoothing_factor=0.1,
    # predict_with_generate=True,
    logging_dir="logs",
    logging_steps=50,
    save_total_limit=3,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_data,
    eval_dataset=validation_data,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# trainer.train()
print("Training Completed")
print("="*100)
# trainer.evaluate()
print("Evaluation completed")
print("="*100)
trainer = AutoModelForSeq2SeqLM.from_pretrained("./Models/"+save_name).to(device)
from tqdm import tqdm

test_targets = []
test_generated = []
i = 0
for idx in tqdm(range(len(test_dataset))):
    inputs = str(test_dataset[idx]['input'])
    input_ids = tokenizer.encode(inputs, return_tensors="pt").to(device)
    output = trainer.generate(input_ids, max_length=50, num_beams=4, early_stopping=True)
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    test_generated.append(output_text)
    test_targets.append(test_dataset[idx]['output'])

import evaluate
rouge = evaluate.load('rouge')
results = rouge.compute(predictions=test_generated, references=test_targets)
print(results)

from datasets import load_metric
metric = load_metric("rouge")
results = metric.compute(predictions=test_generated, references=test_targets)
# print(results)
result = list(results.items())
for index in range(len(result)-1):
  type_, scores = result[index]
  precision = scores.mid.precision
  recall = scores.mid.recall
  f1 = scores.mid.fmeasure
  print(str(type_)+" with precision "+str(precision)+" with recall "+str(recall)+" and F1 score "+str(f1))

from bert_score import score as bert_score
Ptestft, Rtestft, F1testft = bert_score(cands=test_generated, refs=test_targets, lang='en', verbose=True)
print("BERTScore Precision:", Ptestft.mean().item())
print("BERTScore Recall:", Rtestft.mean().item())
print("BERTScore F1:", F1testft.mean().item())