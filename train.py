import pytorch_pretrained_bert
import tokenization
import torch
import numpy as np
import os
import json
import re
import random
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


DATA_PATH = 'small_train_data.json'
tokenizer = tokenization.BasicTokenizer()
full_tokenizer = tokenization.BertTokenizer.from_pretrained('bert-base-chinese', cache_dir='./cache')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = pytorch_pretrained_bert.modeling.BertForQuestionAnswering.from_pretrained('bert-base-chinese')
model.to(device)

EPOCHS = 5
BATCH_SIZE = 12
LR = 1e-5


class FYExample(object):
    def __init__(self,
                 question_tokens,
                 doc_tokens,
                 start_position=None,
                 end_position=None,
                 answer_text=None):
        """

        无法回答：start = 1， end = 0
        yes： start = 2， end = 1
        no： start = 3， end = 2
        """
        self.question_tokens = question_tokens
        self.doc_tokens = doc_tokens
        self.start_position = start_position
        self.end_position = end_position
        self.answer_text = None


class FYFeatures(object):
    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 start_position,
                 end_position):
        """

        无法回答：start = 1， end = 0
        yes： start = 2， end = 1
        no： start = 3， end = 2
        """
        self.input_ids = input_ids,
        self.input_mask = input_mask,
        self.segment_ids = segment_ids,
        self.start_position = start_position,
        self.end_position = end_position,


# class FYDataset(Dataset):
#
#     def __init__(self, features):
#         self.features = features
#
#     def __len__(self):
#         return len(self.features)
#
#     def __getitem__(self, idx):
#         sample = dict()
#         sample['input_ids'] = self.features[idx].input_ids[0]
#         sample['input_mask'] = self.features[idx].input_mask[0]
#         sample['segment_ids'] = self.features[idx].segment_ids[0]
#         sample['start_position'] = self.features[idx].start_position[0]
#         sample['end_position'] = self.features[idx].end_position[0]
#         return sample


def get_data():
    f = open(DATA_PATH, 'r')
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
            if is_impossible != 'false':
                example = FYExample(question_tokens=tokenizer.tokenize(question),
                                    doc_tokens=tokenizer.tokenize(context),
                                    start_position=1,
                                    end_position=0)
                examples.append(example)
            else:
                for answer in answers:
                    if answer['text'] == 'YES':
                        example = FYExample(question_tokens=tokenizer.tokenize(question),
                                            doc_tokens=tokenizer.tokenize(context),
                                            start_position=2,
                                            end_position=1)
                        examples.append(example)
                    elif answer['text'] == 'NO':
                        example = FYExample(question_tokens=tokenizer.tokenize(question),
                                            doc_tokens=tokenizer.tokenize(context),
                                            start_position=3,
                                            end_position=2)
                        examples.append(example)
                    else:
                        tokenized_context = tokenizer.tokenize(context)
                        context_position = 0
                        tokenized_position = 0
                        tokenized_start = 0
                        tokenized_end = 0
                        while context_position < len(context):
                            if answer['answer_start'] == context_position:
                                tokenized_start = tokenized_position
                            if answer['answer_start'] + len(answer['text']) == context_position:
                                tokenized_end = tokenized_position
                                break
                            try:
                                context_position += len(tokenized_context[tokenized_position])
                            except Exception as e:
                                print('exception!')
                                break
                            tokenized_position += 1
                        if tokenized_start <= tokenized_end:
                            example = FYExample(question_tokens=tokenizer.tokenize(question),
                                                doc_tokens=tokenized_context,
                                                start_position=tokenized_start,
                                                end_position=tokenized_end,
                                                answer_text=tokenizer.tokenize(answer['text']))
                            # print()
                            # print(''.join(tokenized_context))
                            # print(''.join(question))
                            # print(''.join(tokenized_context[tokenized_start:tokenized_end]))
                            # print()
                            examples.append(example)
    f.close()
    return examples


def convert_examples_to_features(examples):
    features = []
    for example in examples:
        question = example.question_tokens
        doc = example.doc_tokens
        start_position = example.start_position
        end_position = example.end_position
        if len(question) + len(doc) + 3 <= 512:
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
                                 start_position=start_position + len(question) + 2,
                                 end_position=end_position + len(question) + 2)
            # print()
            # print(full_tokenizer.convert_ids_to_tokens(input_ids))
            # print(full_tokenizer.convert_ids_to_tokens(input_ids[start_position + len(question) + 2: end_position + len(question) + 2]))
            features.append(feature)
        else:
            start_point = 0
            while start_point < start_position:
                tokens = []
                segment_ids = []
                tokens.append("[CLS]")
                segment_ids.append(0)
                for token in question:
                    tokens.append(token)
                    segment_ids.append(0)
                tokens.append("[SEP]")
                segment_ids.append(0)
                for token in doc[start_point:start_point + (512 - 3 - len(question))]:
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
                                     start_position=start_position + len(question) + 2 - start_point,
                                     end_position=end_position + len(question) + 2 - start_point)
                # print()
                # print(full_tokenizer.convert_ids_to_tokens(input_ids))
                # print(full_tokenizer.convert_ids_to_tokens(
                #     input_ids[start_position + len(question) + 2 - start_point: end_position + len(question) + 2 - start_point]))
                features.append(feature)
                start_point += 125
    return features


def main():
    examples = get_data()
    features = convert_examples_to_features(examples)
    print('length of features', len(features))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    running_loss = 0
    for epoch in range(EPOCHS):
        random.shuffle(features)
        for step in range(len(features) // BATCH_SIZE):
            batch = features[step * BATCH_SIZE: (step + 1) * BATCH_SIZE]
            input_ids = torch.Tensor([feature.input_ids for feature in batch]).squeeze(1).long().to(device)
            token_type_ids = torch.Tensor([feature.segment_ids for feature in batch]).squeeze(1).long().to(device)
            input_mask = torch.Tensor([feature.input_mask for feature in batch]).squeeze(1).long().to(device)
            start_positions = torch.Tensor([feature.start_position for feature in batch]).squeeze(1).long().to(device)
            pend_positions = torch.Tensor([feature.end_position for feature in batch]).squeeze(1).long().to(device)

            optimizer.zero_grad()
            loss = model(input_ids=input_ids, token_type_ids=token_type_ids,
                         attention_mask=input_mask, start_positions=start_positions,
                         end_positions=pend_positions)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (step + 1) % 200 == 0:
                print('step {} of epoch {}, loss {}'.format(step + 1, epoch + 1, running_loss / 200))
                running_loss = 0.0


if __name__ == '__main__':
    main()
