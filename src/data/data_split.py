import gc
import multiprocessing

import gensim.downloader as api
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from gensim.models import Word2Vec, FastText
from sklearn.model_selection import train_test_split
from torch.utils.data import (TensorDataset, DataLoader, RandomSampler)
from torchtext.data.utils import get_tokenizer
from transformers import BertTokenizer, BertModel


class DataSplitModule:
    def __init__(
            self,
            dataset_name,
            dataset_dir
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name

    def split_and_save_data(self):
        dataset = load_dataset(f"Authorship/{self.dataset_name}", data_dir=self.dataset_dir)
        dataset = pd.concat(
            [pd.DataFrame(dataset['train']), pd.DataFrame(dataset['test']), pd.DataFrame(dataset['validation'])])

        max_length = dataset['BookText'].str.len().max()
        dataset['BookText'] = dataset['BookText'].apply(lambda x: x + ' ' * (max_length - len(x)))

        label_list = dataset['AuthorID'].unique().tolist()

        label_map = {}
        for (i, label) in enumerate(label_list):
            label_map[label] = i

        dataset = dataset.sample(frac=1, random_state=42)

        dataset['label'] = dataset['AuthorID'].apply(lambda id: label_map[id])

        trainval, test = train_test_split(
            dataset, test_size=0.2, stratify=dataset['label'], random_state=42)

        trainval.to_csv(f'/content/ChatGPT/{self.dataset_dir}_train_val.csv', index=False)

        test.to_csv(f'/content/ChatGPT/{self.dataset_dir}_test.csv', index = False)

    


