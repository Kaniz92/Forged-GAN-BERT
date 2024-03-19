import pandas as pd
import torch
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import (TensorDataset, DataLoader, RandomSampler)

from src.models.data_module import DataModule
import gc


class TransformerDataModule(DataModule):
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
        self.tokenizer = self.tokenizer_class.from_pretrained(self.tokenizer_name,
                                                              do_lower_case=self.do_lower_case)

        train_val_dataset = pd.DataFrame(load_dataset(f"Authorship/{self.dataset_name}", data_files=[f'{self.dataset_dir}_train_val.csv'])['train'])
        test_dataset = pd.DataFrame(load_dataset(f"Authorship/{self.dataset_name}", data_files=[f'{self.dataset_dir}_test.csv'])['train'])

        train_val_split = self.get_split_data(train_val_dataset)
        test_split = self.get_split_data(test_dataset, isTrain = False)

        # If zero-shot testing, we pass the whole dataset
        if self.training_strategy == 'best_modal_on_wandb_RQ4':
            combined_dataset = pd.concat([train_val_split, test_split])
            _, _, self.test_dataset = self.generate_transformer_dataloader(combined_dataset)
        else:
            self.data_all_inputs, self.data_all_labels, self.train_val_dataset = self.generate_transformer_dataloader(
                train_val_split)
            _, _, self.test_dataset = self.generate_transformer_dataloader(test_split)

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

    def generate_bert_inputs(self, document):
        # Tokenize the document
        tokens = self.tokenizer.tokenize(document)

        # Set the maximum window size and stride
        window_size = 512
        stride = 256

        # Initialize lists to store input IDs and attention masks
        input_ids_list = []
        attention_mask_list = []

        # Process the document using a sliding window
        start = 0
        while start < len(tokens):
            end = min(start + window_size, len(tokens))

            window_tokens = tokens[start:end]

            input_ids = self.tokenizer.convert_tokens_to_ids(window_tokens)
            attention_mask = [1] * len(input_ids)

            if len(input_ids) > window_size:
                num_tokens = len(input_ids)
                stride = num_tokens // window_size
                aggregated_input_ids = [input_ids[i] for i in range(0, num_tokens, stride)]
                input_ids = aggregated_input_ids[:window_size]

            padding_length = window_size - len(input_ids)
            input_ids += [0] * padding_length
            attention_mask += [0] * padding_length

            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)

            start += stride

        input_ids = torch.tensor(input_ids_list, dtype=torch.long).to('cpu')
        attention_mask = torch.tensor(attention_mask_list).to('cpu')

        input_ids_mean = torch.mean(input_ids.float(), dim=0)
        attention_mask_mean = torch.mean(attention_mask.float(), dim=0)

        return input_ids_mean.long(), attention_mask_mean.long()

    def generate_transformer_dataloader(self, input_examples):
        text_a, labels = (
            input_examples["BookText"].astype(str).tolist(),
            input_examples["AuthorID"].tolist(),
        )

        labels = [self.label_map[label] for label in labels]

        input_id_list = []
        attention_mask_list = []

        for document in text_a:
            input_ids, attention_mask = self.generate_bert_inputs(document)
            input_id_list.append(input_ids)
            attention_mask_list.append(attention_mask)

        labels = torch.tensor(labels, dtype=torch.long)
        input_ids_list = torch.stack(input_id_list, dim=0)
        attention_mask_list = torch.stack(attention_mask_list, dim=0)

        dataset = TensorDataset(input_ids_list, attention_mask_list, labels)

        return input_ids_list, labels, dataset
