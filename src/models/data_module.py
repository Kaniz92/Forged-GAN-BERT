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


class DataModule:
    def __init__(
            self,
            dataset_name,
            dataset_dir,
            batch_size,
            training_strategy=None,
            apply_balance=None,
            embedding_model_name=None,
            tokenizer_name=None,
            do_lower_case=None,
            tokenizer_class=None,
            embedding_class=None
    ):
        super().__init__()
        self.embed_dim = None
        self.vocab_size = None
        self.data_all_labels = None
        self.data_all_inputs = None
        self.test_labels = None
        self.test_inputs = None
        self.embedding_matrix = None
        self.label_map = None
        self.tokenizer = None
        self.embeddings = None
        self.label_list = None
        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.training_strategy = training_strategy
        self.embedding_model_name = embedding_model_name
        self.apply_balance = apply_balance
        self.tokenizer_class = tokenizer_class
        self.tokenizer_name = tokenizer_name
        self.do_lower_case = do_lower_case
        self.embedding_class = embedding_class

        self.max_seq_length = 512

        self.embedding_methods = {
            'Word2Vec': self.word2vec_embeddings,
            'GloVe': self.glove_embeddings,
            'FastText': self.fasttext_embeddings,
            'BERT': self.bert_embeddings
        }

        self.prepare_data()

    def word2vec_embeddings(self, preprocessed_documents):
        cores = multiprocessing.cpu_count()
        w2v_model = Word2Vec(preprocessed_documents, min_count=20, window=2, sample=6e-5, alpha=0.03, min_alpha=0.0007,
                             negative=20, workers=cores - 1)
        w2v_model.train(preprocessed_documents, total_examples=w2v_model.corpus_count, epochs=100)

        document_embeddings = []
        for document in preprocessed_documents:
            word_embeddings = [w2v_model.wv[word] for word in document if word in w2v_model.wv]
            if word_embeddings:
                document_embedding = np.mean(word_embeddings, axis=0)
            else:
                document_embedding = np.zeros(w2v_model.vector_size)
            document_embeddings.append(document_embedding)

        input_ids = []
        for document_embedding in document_embeddings:
            word_indices = [w2v_model.wv.key_to_index[word] for word in document if word in w2v_model.wv.key_to_index]
            input_ids.append(torch.LongTensor(word_indices))

        padded_input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True)
        input_ids_array = padded_input_ids.numpy()

        self.embedding_matrix = w2v_model.wv.vectors
        self.embeddings = input_ids_array
        self.vocab_size = len(w2v_model.wv.key_to_index)
        self.embed_dim = w2v_model.vector_size

    def glove_embeddings(self, preprocessed_documents):
        glove_model = api.load("glove-wiki-gigaword-50")

        document_embeddings = []
        for document in preprocessed_documents:
            word_embeddings = [glove_model[word] for word in document if word in glove_model]
            if word_embeddings:
                document_embedding = np.mean(word_embeddings, axis=0)
            else:
                document_embedding = np.zeros(glove_model.vector_size)
            document_embeddings.append(document_embedding)

        self.embedding_matrix = glove_model.vectors
        self.embeddings = document_embeddings
        self.vocab_size = len(glove_model.key_to_index)
        self.embed_dim = glove_model.vector_size

    def fasttext_embeddings(self, preprocessed_documents):
        cores = multiprocessing.cpu_count()
        fasttext_model = FastText(preprocessed_documents, min_count=20, window=2, sample=6e-5, alpha=0.03,
                                  min_alpha=0.0007,
                                  negative=20, workers=cores - 1)
        fasttext_model.train(preprocessed_documents, total_examples=fasttext_model.corpus_count, epochs=100)

        document_embeddings = []
        for document in preprocessed_documents:
            word_embeddings = [fasttext_model.wv[word] for word in document if word in fasttext_model.wv]
            if word_embeddings:
                document_embedding = np.mean(word_embeddings, axis=0)  # Average the word embeddings
            else:
                document_embedding = np.zeros(fasttext_model.vector_size)  # Use zero vector if no embeddings found
            document_embeddings.append(document_embedding)

        self.embedding_matrix = fasttext_model.wv.vectors
        self.embeddings = document_embeddings
        self.vocab_size = len(fasttext_model.wv.key_to_index)
        self.embed_dim = fasttext_model.vector_size

    def generate_bert_inputs(self, document):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # Tokenize the document
        tokens = self.tokenizer.tokenize(document)

        # Set the maximum window size and stride
        window_size = 256
        stride = 512

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

    def bert_embeddings(self, preprocessed_documents):
        model = BertModel.from_pretrained('bert-base-uncased')

        input_id_list = []
        attention_mask_list = []

        for document in preprocessed_documents:
            input_ids, attention_mask = self.generate_bert_inputs(document)
            input_id_list.append(input_ids)
            attention_mask_list.append(attention_mask)

        labels = torch.tensor(labels, dtype=torch.long)
        input_ids_list = torch.stack(input_id_list, dim=0)
        attention_mask_list = torch.stack(attention_mask_list, dim=0)

        with torch.no_grad():
            outputs = model(input_ids_list, attention_mask=attention_mask_list)

        self.embeddings = outputs.last_hidden_state

        num_documents, max_seq_length, embedding_dim = self.embeddings.shape
        self.embeddings = self.embeddings.reshape(num_documents, max_seq_length * embedding_dim)

        self.vocab_size = self.tokenizer.vocab_size
        self.embed_dim = self.embeddings.shape  # [2]

        self.embedding_matrix = model.embeddings.word_embeddings.weight

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

        self.label_map = {}
        for (i, label) in enumerate(self.label_list):
            self.label_map[label] = i

        dataset = dataset.sample(frac=1, random_state=42)

        preprocessed_documents = np.array(dataset['BookText'])

        dataset['label'] = dataset['AuthorID'].apply(lambda id: self.label_map[id])
        labels = np.array(dataset['label'])

        get_embedding = self.embedding_methods[self.embedding_class]
        get_embedding(preprocessed_documents)
        inputs = np.array(self.embeddings)

        return inputs, labels

    def prepare_data(self):
        self.tokenizer = get_tokenizer("basic_english")

        self.train_val_dataset = pd.DataFrame(load_dataset(f"Authorship/{self.dataset_name}", data_files=[f'{self.dataset_dir}_train_val.csv'])['train'])
        self.test_dataset = pd.DataFrame(load_dataset(f"Authorship/{self.dataset_name}", data_files=[f'{self.dataset_dir}_test.csv'])['train'])

        self.data_all_inputs, self.data_all_labels = self.get_split_data(self.train_val_dataset)
        self.test_inputs, self.test_labels = self.get_split_data(self.test_dataset, isTrain = False)

        del self.train_val_dataset, self.test_dataset
        gc.collect()

    def get_data_loaders(self, train_index, test_index):
        tensor_inputs, tensor_labels = tuple(torch.tensor(data) for data in
                                             [self.data_all_inputs, self.data_all_labels])
        dataset = TensorDataset(tensor_inputs, tensor_labels)

        collate_fn = None

        train_sampler = RandomSampler(train_index)
        train_dataloader = DataLoader(dataset, batch_size=self.batch_size, sampler=train_sampler,
                                      collate_fn=collate_fn)
        test_sampler = RandomSampler(test_index)
        test_dataloader = DataLoader(dataset, sampler=test_sampler, batch_size=self.batch_size,
                                     collate_fn=collate_fn)

        return train_dataloader, test_dataloader

    def get_test_dataloader(self):
        tensor_inputs, tensor_labels = tuple(torch.tensor(data) for data in
                                             [self.test_inputs, self.test_labels])
        dataset = TensorDataset(tensor_inputs, tensor_labels)
        collate_fn = None

        test_sampler = RandomSampler(dataset)
        test_dataloader = DataLoader(dataset, batch_size=self.batch_size, sampler=test_sampler,
                                     collate_fn=collate_fn)
        return test_dataloader

    def get_pos_label(self):
        # TODO: verify this for GAN-BERT
        return len(self.label_list) - 1

    def get_embedding_matrix(self):
        return self.embedding_matrix

    def get_data_all(self):
        return self.data_all_inputs, self.data_all_labels

    def get_test_data(self):
        return self.test_inputs, self.test_labels

    def get_label_list(self):
        return self.label_list

    def get_embeddings(self):
        return self.vocab_size, self.embed_dim, self.embedding_matrix
