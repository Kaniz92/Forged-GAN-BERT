import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import (TensorDataset, DataLoader, RandomSampler)
from transformers import *
import gc

from src.models.data_module import DataModule


class GANBertDataModule(DataModule):
    def get_original_author(self, row):
        if row['AuthorID'] == 9999:
          orginal_book_title = row['BookTitle'].split('_')[1]
          original_book = self.train_val_dataset[self.train_val_dataset['BookTitle'] == orginal_book_title]
          
          if len(original_book) == 0:
              original_book = self.test_dataset[self.test_dataset['BookTitle'] == orginal_book_title]
          
          row['AuthorID'] = original_book.iloc[0]['AuthorID']

        return row['AuthorID']

    def get_split_data(self, dataset, isTrain = True):
        if self.training_strategy == 'best_modal_on_wandb_RQ3':
            dataset['AuthorID'] = dataset['AuthorID'].apply(lambda x: 1 if x == 9999 else 0)

        if self.training_strategy == 'best_modal_on_wandb_RQ5':
            # here we only use human novels to train the models and then test with chatgpt novels
            # here the dataset is lesser but consistent across all models with same examples during train and test phases
            if isTrain:
                dataset = dataset[dataset['AuthorID'] != 9999]
                self.label_list = dataset['AuthorID'].unique().tolist()
                self.label_list.append(9999)
                
            else:
                dataset['AuthorID'] = dataset.apply(lambda row: self.get_original_author(row), axis=1)

        self.label_list = dataset['AuthorID'].unique().tolist()

        self.label_map = {}
        for (i, label) in enumerate(self.label_list):
            self.label_map[label] = i

        dataset = dataset.sample(frac=1, random_state=42)

        dataset['label'] = dataset['AuthorID'].apply(lambda id: self.label_map[id])

        return dataset

    def prepare_data(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_name)

        train_val_dataset = pd.DataFrame(load_dataset(f"Authorship/{self.dataset_name}", data_files=[f'{self.dataset_dir}_train_val.csv'])['train'])
        test_dataset = pd.DataFrame(load_dataset(f"Authorship/{self.dataset_name}", data_files=[f'{self.dataset_dir}_test.csv'])['train'])

        train_val_split = self.get_split_data(train_val_dataset)
        test_split = self.get_split_data(test_dataset, isTrain = False)

        label_masks = np.ones(len(train_val_split), dtype=bool)
        self.data_all_inputs, self.data_all_labels, self.train_val_dataset = self.generate_GAN_BERT_dataloader(
            train_val_split, label_masks, self.label_map, balance_label_examples=self.apply_balance)

        test_label_masks = np.ones(len(test_split), dtype=bool)
        _, _, self.test_dataset = self.generate_GAN_BERT_dataloader(test_split, test_label_masks, self.label_map,
                                                                    balance_label_examples=self.apply_balance)

        del train_val_dataset, test_dataset, train_val_split, test_split
        gc.collect()

    def get_data_loaders(self, train_index, test_index):
        dataset = self.train_val_dataset

        collate_fn = None

        train_sampler = RandomSampler(train_index)
        train_dataloader = DataLoader(dataset, batch_size=self.batch_size, sampler=train_sampler,
                                      collate_fn=collate_fn)
        test_sampler = RandomSampler(test_index)
        test_dataloader = DataLoader(dataset, sampler=test_sampler, batch_size=self.batch_size,
                                     collate_fn=collate_fn)

        return train_dataloader, test_dataloader

    def get_test_dataloader(self):
        dataset = self.test_dataset
        collate_fn = None

        test_sampler = RandomSampler(dataset)
        test_dataloader = DataLoader(dataset, batch_size=self.batch_size, sampler=test_sampler,
                                     collate_fn=collate_fn)
        return test_dataloader

    def generate_GAN_BERT_dataloader(self, input_examples, label_masks, label_map, balance_label_examples=False):
        examples = []

        num_labeled_examples = 0
        for label_mask in label_masks:
            if label_mask:
                num_labeled_examples += 1
        label_mask_rate = num_labeled_examples / len(input_examples)

        # if required it applies the balance
        input_examples = input_examples.reset_index(drop=True)
        for index, ex in input_examples.iterrows():
            if label_mask_rate == 1 or not balance_label_examples:
                examples.append((ex, label_masks[index]))
            else:
                # IT SIMULATE A LABELED EXAMPLE
                if label_masks[index]:
                    balance = int(1 / label_mask_rate)
                    balance = int(math.log(balance, 2))
                    if balance < 1:
                        balance = 1
                    for b in range(0, int(balance)):
                        examples.append((ex, label_masks[index]))
                else:
                    examples.append((ex, label_masks[index]))

        input_ids = []
        input_mask_array = []
        label_mask_array = []
        label_id_array = []

        # Tokenization
        for (text, label_mask) in examples:
            encoded_sent = self.tokenizer.encode(text['BookText'], add_special_tokens=True,
                                                 max_length=self.max_seq_length, padding="max_length", truncation=True)
            input_ids.append(encoded_sent)
            label_id_array.append(label_map[text['AuthorID']])
            label_mask_array.append(label_mask)

        # Attention to token (to ignore padded input wordpieces)
        for sent in input_ids:
            att_mask = [int(token_id > 0) for token_id in sent]
            input_mask_array.append(att_mask)
        # Convertion to Tensor
        input_ids = torch.tensor(input_ids)
        input_mask_array = torch.tensor(input_mask_array)
        label_id_array = torch.tensor(label_id_array, dtype=torch.long)
        label_mask_array = torch.tensor(label_mask_array)

        # Building the TensorDataset
        dataset = TensorDataset(input_ids, input_mask_array, label_id_array, label_mask_array)

        return input_ids, label_id_array, dataset
