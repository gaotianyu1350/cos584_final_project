import argparse
import os
import numpy as np
from tqdm import tqdm
import pandas as pd
from pandas import DataFrame
import nltk
import random
import shutil
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5ForConditionalGeneration
import transformers
import torch


import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas

parser = argparse.ArgumentParser()

parser.add_argument("--k", type=int, help="Number of training instances per label", default=16)
parser.add_argument("--data_dir", type=str, default=".", help="Path to few-shot data")
parser.add_argument("--seed", type=int, nargs="+", default=[42, 13, 21, 87, 100], help="Seeds for data splits")
parser.add_argument("--task", type=str, nargs="+", default=["SST-2", "sst-5", "mr", "cr", "mpqa", "subj", "trec", "CoLA", "MRPC", "QQP", "STS-B", "MNLI", "SNLI", "QNLI", "RTE"], help="Tasks")
parser.add_argument("--aug", type=str)
parser.add_argument("--aug_num", type=int, default=1)
parser.add_argument("--aug_rate", type=float, default=0.1)
parser.add_argument("--aug_name", type=str)
parser.add_argument("--ft", action="store_true")

args = parser.parse_args()

global model
if args.aug == "t5":
    tokenizer = AutoTokenizer.from_pretrained("google/t5-v1_1-large")
    model = None
elif args.aug == "synonym":
    aug_model = naw.SynonymAug(aug_src='wordnet')
elif args.aug == "synonym_ppdb":
    aug_model = naw.SynonymAug(aug_src='ppdb', model_path="ppdb-2.0-tldr")
else:
    raise NotImplementedError

def t5_generated_result(sent):
    input_ids = tokenizer(sent, return_tensors="pt").input_ids
    input_ids = input_ids.cuda()
    output_ids = model.generate(input_ids, num_beams=3)
    output = tokenizer.decode(output_ids[0])
    output = output.split('<extra_id_0>')[1].split('<extra_id_1>')[0]
    print("T5 output:", output)
    return output

def get_sentence(task, line):
    if task in ['mr', 'sst-5', 'subj', 'trec', 'cr', 'mpqa']:
        # Text classification tasks
        if line[1] is None or pd.isna(line[1]):
            return ['']
        else:
            return [line[1]]
    else:
        # GLUE tasks
        line = line.strip().split('\t')
        if task == 'CoLA':
            return [line[-1]]
        elif task == 'MNLI':
            return [line[8], line[9]]
        elif task == 'MRPC':
            return [line[-2], line[-1]]
        elif task == 'QNLI':
            return [line[1], line[2]]
        elif task == 'QQP':
            return [line[3], line[4]]
        elif task == 'RTE':
            return [line[1], line[2]]
        elif task == 'SNLI':
            return [line[7], line[8]]
        elif task == 'SST-2':
            return [line[0]]
        elif task == 'STS-B':
            return [line[-3], line[-2]]
        elif task == 'WNLI':
            return [line[1], line[2]]
        else:
            raise NotImplementedError

def feed_sentence(task, line, sent):
    if task in ['mr', 'sst-5', 'subj', 'trec', 'cr', 'mpqa']:
        # Text classification tasks
        return line[:1] + sent + line[2:]
    else:
        # GLUE tasks
        line = line.strip().split('\t')
        if task == 'CoLA':
            return line[:-1] + sent
        elif task == 'MNLI':
            return line[:8] + sent + line[10:]
        elif task == 'MRPC':
            return line[:-2] + sent
        elif task == 'QNLI':
            return line[:1] + sent + line[3:]
        elif task == 'QQP':
            return line[:3] + sent + line[5:]
        elif task == 'RTE':
            return line[:1] + sent + line[3:]
        elif task == 'SNLI':
            return line[:7] + sent + line[9:]
        elif task == 'SST-2':
            return sent + line[1:]
        elif task == 'STS-B':
            return line[:-3] + sent + line[-1:]
        elif task == 'WNLI':
            return line[:1] + sent + line[3:]
        else:
            raise NotImplementedError

def split_header(task, lines):
    """Returns if the task file has a header or not."""
    if task in ["CoLA"]:
        return [], lines
    elif task in ["MNLI", "MRPC", "QNLI", "QQP", "RTE", "SNLI", "SST-2", "STS-B", "WNLI"]:
        return lines[0:1], lines[1:]
    else:
        raise ValueError("Unknown GLUE task.")

def load_datasets(data_dir, task):
    dataset = {}
    header = None
    splits = ["train"]
    for split in splits:
        if task in ['mr', 'sst-5', 'subj', 'trec', 'cr', 'mpqa']:
            filename = os.path.join(data_dir, f"{split}.csv")
            dataset[split] = pd.read_csv(filename, header=None).values.tolist()
        else:
            filename = os.path.join(data_dir, f"{split}.tsv")
            with open(filename, "r") as f:
                lines = f.readlines()
                header, content = split_header(task, lines)
            dataset[split] = content
    return header, dataset

def save_new_datasets(data_dir, task, header, new_dataset):
    splits = ["train"]
    for split in splits:
        if task in ['mr', 'sst-5', 'subj', 'trec', 'cr', 'mpqa']:
            filename = os.path.join(data_dir, f"{split}.csv")
            DataFrame(new_dataset[split]).to_csv(filename, header=False, index=False)
        else:
            filename = os.path.join(data_dir, f"{split}.tsv")
            with open(filename, "w") as f:
                for line in header:
                    f.write(line.rstrip() + '\n')
                for line in new_dataset[split]:
                    f.write(line + '\n')

def aug(sents, args):
    new_sets_sents = []
    for _ in range(args.aug_num): # How many times we augment the data
        new_sents = []
        for sent in sents:
            if args.aug == "t5":
                tokens = sent.split(' ') 
                l = min(max(int(len(tokens) * args.aug_rate), 1), len(tokens) - 1)
                start = random.randint(0, len(tokens) - l)
                end = start + l

                print("##################")
                input_sent = ' '.join(tokens[:start]) + "<extra_id_0> " + ' '.join(tokens[end:])
                output = t5_generated_result(input_sent)
                new_sent = ' '.join(tokens[:start]) + output + ' ' + ' '.join(tokens[end:])
                print("Original sent:", sent)
                print("Masked sent:", input_sent)
                print("New sent:", new_sent)
                new_sents.append(new_sent)
            else:
                print("##################")
                new_sent = aug_model.augment(sent)
                print("Original sent:", sent)
                print("New sent:", new_sent)
                new_sents.append(new_sent)
        new_sets_sents.append(new_sents)
    return new_sets_sents
    
def main():

    for task in args.task:
        for seed in args.seed:
            folder = os.path.join(args.data_dir, task, '{}-{}'.format(args.k, seed))
            new_folder = os.path.join(args.data_dir, task + "-" + args.aug_name, '{}-{}'.format(args.k, seed))
            os.makedirs(new_folder, exist_ok=True)
            os.system("cp -r " + os.path.join(folder, "train*") + " " + new_folder)        
            os.system("cp -r " + os.path.join(folder, "dev*") + " " + new_folder)        
            os.system("cp -r " + os.path.join(folder, "test*") + " " + new_folder)        
            
            header, dataset = load_datasets(folder, task)
            new_dataset = {}
            for split in dataset:
                print('{}-{}-{}-{}'.format(task, args.k, seed, split))
                lines = dataset[split]

                # train new model
                training_set = []
                for line in lines:
                    sents = get_sentence(task, line)
                    training_set += sents
                print("total sent:", len(training_set))

                # Train
                print("Load T5 model")
                global model

                model = T5ForConditionalGeneration.from_pretrained("google/t5-v1_1-large")
                model = model.cuda()
                model.train()
                decay_parameters=['bias', 'LayerNorm.bias', 'LayerNorm.weight']
                grouped_parameters = [
                    {
                        "params": [p for n, p in model.named_parameters() if n in decay_parameters],
                        "weight_decay": 0.01,
                        "lr": 1e-5
                    },
                    {
                        "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
                        "weight_decay": 0.0,
                        "lr": 1e-5
                    },
                ]
                optimizer = transformers.AdamW(grouped_parameters)
                
                num_epochs = 10
                batch_size = 8
                for epoch_id in range(num_epochs):
                    total_batch = len(training_set) // batch_size
                    for batch_id in range(total_batch):
                        ori_batch_sent = training_set[batch_id*batch_size:(batch_id+1)*batch_size]
                        batch_sent = []
                        batch_label = []
                        for sent in ori_batch_sent:
                            tokens = sent.split(' ') 
                            l = min(max(int(len(tokens) * args.aug_rate), 1), len(tokens) - 1)
                            start = random.randint(0, len(tokens) - l)
                            end = start + l
                            new_sent = ' '.join(tokens[:start]) + "<extra_id_0> " + ' '.join(tokens[end:])
                            new_label = "<extra_id_0>" + '  '.join(tokens[start:end]) + "<extra_id_1>"
                            # print("######")
                            # print(new_sent)
                            # print(new_label)

                            batch_sent.append(new_sent)
                            batch_label.append(new_label)

                        input_batch = tokenizer(batch_sent, return_tensors="pt", padding=True, truncation=True, max_length=32)
                        decoder_input_batch = tokenizer(batch_label, return_tensors="pt", padding=True, truncation=True, max_length=32)
                        labels = decoder_input_batch["input_ids"]
                        tmp = torch.zeros(decoder_input_batch["input_ids"].size()).long()
                        tmp[:, 1:] = decoder_input_batch["input_ids"][:, :-1]
                        decoder_input_batch["input_ids"] = tmp

                        input_batch = {k: v.cuda() for k, v in input_batch.items()}
                        decoder_input_batch = {"decoder_" + k: v.cuda() for k, v in decoder_input_batch.items()}
                        labels = labels.cuda()

                        output = model(**input_batch, **decoder_input_batch, labels=labels, return_dict=True)
                        loss = output.loss
                        print(loss.item())

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                
                print("finish training")
                model.eval()

                new_lines = []
                for line in lines:
                    sents = get_sentence(task, line)
                    new_sets_sents = aug(sents, args)
                    if isinstance(line, list):
                        new_lines.append(line)
                    else:
                        new_lines.append(line.rstrip())
                    for new_sents in new_sets_sents:
                        if isinstance(line, list):
                            new_line = feed_sentence(task, line, new_sents)
                        else:
                            new_line = '\t'.join(feed_sentence(task, line, new_sents))
                        new_lines.append(new_line)
                new_dataset[split] = new_lines 
            save_new_datasets(new_folder, task, header, new_dataset)

if __name__ == '__main__':
    main()
