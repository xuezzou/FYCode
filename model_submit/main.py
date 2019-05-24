# coding: utf-8
import pytorch_pretrained_bert
import tokenization
import torch
import numpy as np
import os
import json
import re
import random
import pickle
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

infile = "../data/data.json"
outfile = "../result/result.json"
tokenizer = tokenization.BasicTokenizer()
full_tokenizer = tokenization.BertTokenizer.from_pretrained('bert-base-chinese', cache_dir='./cache')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.load('./model.pt')
model.to(device)


class FYExample(object):
    def __init__(self,
                 question_tokens,
                 doc_tokens,
                 start_position=None,
                 end_position=None,
                 answer_text=None,
                 id=None):
        """
        tokenized完的例子
        无法回答：start = 1， end = 0
        yes： start = 2， end = 1
        no： start = 3， end = 2
        """
        self.question_tokens = question_tokens
        self.doc_tokens = doc_tokens
        self.start_position = start_position
        self.end_position = end_position
        self.answer_text = answer_text
        self.id = id


class FYFeatures(object):
    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 start_position=None,
                 end_position=None,
                 id=None):
        """
        变成输入到模型中的features的例子
        无法回答：start = 1， end = 0
        yes： start = 2， end = 1
        no： start = 3， end = 2
        """
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position
        self.id = id


def get_data(data):
    f = data
    jf = json.load(f)
    cases = jf['data']
    paragraphs = []
    for case in cases:
        for paragraph in case['paragraphs']:
            paragraphs.append(paragraph)
    examples = []
    for paragraph in tqdm(paragraphs):
        context = paragraph['context']
        qas = paragraph['qas']
        for qa in qas:
            question = qa['question']
            answers = qa['answers']
            is_impossible = qa['is_impossible']
            id = qa['id']
            example = FYExample(question_tokens=tokenizer.tokenize(question), doc_tokens=tokenizer.tokenize(context),
                                id=id)
            examples.append(example)
    f.close()
    return examples


def convert_examples_to_features(examples):
    features = []
    for example in examples:
        question = example.question_tokens
        doc = example.doc_tokens

        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in question:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)
        for token in doc:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)
        input_ids = full_tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        while len(input_ids) < 512:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
        feature = FYFeatures(input_ids=input_ids,
                             input_mask=input_mask,
                             segment_ids=segment_ids,
                             id=example.id)
        # print()
        # print(full_tokenizer.convert_ids_to_tokens(input_ids))
        # print(full_tokenizer.convert_ids_to_tokens(input_ids[start_position + len(question) + 2: end_position + len(question) + 2]))
        features.append(feature)

    return features


def predict(model, features):
    results = []

    for step in range(len(features)):
        batch = [features[step]]

        input_ids = torch.Tensor([feature.input_ids for feature in batch]).long().to(device)
        token_type_ids = torch.Tensor([feature.segment_ids for feature in batch]).long().to(device)
        input_mask = torch.Tensor([feature.input_mask for feature in batch]).long().to(device)
        id = batch[0].id

        start_position_logits, end_position_logits = model(input_ids=input_ids, token_type_ids=token_type_ids,
                                                           attention_mask=input_mask)
        predicted_start_positions = torch.argmax(start_position_logits, dim=-1, keepdim=False)
        predicted_end_positions = torch.argmax(end_position_logits, dim=-1, keepdim=False)

        answer = None
        if predicted_start_positions == 1 and predicted_end_positions == 0:
            answer = ''
        elif predicted_start_positions == 2 and predicted_end_positions == 1:
            answer = 'YES'
        elif predicted_start_positions == 3 and predicted_end_positions == 2:
            answer = 'NO'
        else:
            answer = ''.join(
                full_tokenizer.convert_ids_to_tokens(input_ids)[predicted_start_positions:predicted_end_positions])
        print(answer)
        result = {'id': id,
                  'answer': answer}
        results.append(result)

    return results


def main():
    data = open(infile, "r")
    result = open(outfile, "w")

    examples = get_data(data)
    features = convert_examples_to_features(examples)
    results = predict(model, features)

    json.dump(results, result)


if __name__ == '__main__':
    main()
