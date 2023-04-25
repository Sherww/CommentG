import os
import pandas as pd
import json
import gzip
import tqdm
import pickle
import numpy as np

from torch.utils.data import Dataset, DataLoader
import torch
import transformers
# from codeqg.utils import question_to_id, question_types, chunked
from inlinebart.utils import chunked

class BARTCommentGenerationDataset(Dataset):
    def __init__(self, file_path: str, source_tokenizer, target_tokenizer):
        self.file_path = file_path
        self.dataset = []

        print(f'Reading {file_path}')
        with open(file_path,'rt', encoding='utf-8') as f:
            for line in tqdm.tqdm(f.readlines()):
                self.dataset.append(json.loads(line))

        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.cached_data = None
        self.preprocess_data()

    def preprocess_data(self):
        print(f'Preprocessing {self.file_path}')
        self.cached_data = []

        for chunk_data in tqdm.tqdm(list(chunked(self.dataset, 256))):
            source_ids = self.source_tokenizer([d['code'] for d in chunk_data], max_length=128, truncation=True)['input_ids']
            target_ids = self.target_tokenizer([d['comment'] for d in chunk_data], max_length=32, truncation=True)['input_ids']
            # source_ids = self.source_tokenizer([d['question_type'] + ': ' + d['code'] for d in chunk_data], max_length=64, truncation=True)['input_ids']
            # target_ids = self.target_tokenizer([d['question'] for d in chunk_data], max_length=64, truncation=True)['input_ids']
            for i in range(len(chunk_data)):
                self.cached_data.append((
                    torch.tensor(source_ids[i], dtype=torch.long),
                    torch.tensor(target_ids[i], dtype=torch.long)
                ))

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.cached_data[idx]