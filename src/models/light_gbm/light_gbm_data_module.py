import multiprocessing

import gensim.downloader as api
import numpy as np
import torch
from gensim.models import Word2Vec, FastText
from transformers import BertTokenizer, BertModel

from src.models.data_module import DataModule


class LightGBMDataModule(DataModule):
    def word2vec_embeddings(self, preprocessed_documents):
        cores = multiprocessing.cpu_count()
        w2v_model = Word2Vec(preprocessed_documents, min_count=20, window=2, sample=6e-5, alpha=0.03, min_alpha=0.0007,
                             negative=20, workers=cores - 1)
        w2v_model.train(preprocessed_documents, total_examples=w2v_model.corpus_count, epochs=100)

        document_embeddings = []
        for document in preprocessed_documents:
            word_embeddings = [w2v_model.wv[word] for word in document if word in w2v_model.wv]
            if word_embeddings:
                document_embedding = np.mean(word_embeddings, axis=0)  # Average the word embeddings
            else:
                document_embedding = np.zeros(w2v_model.vector_size)  # Use zero vector if no embeddings found
            document_embeddings.append(document_embedding)

        input_ids = []
        for document_embedding in document_embeddings:
            word_indices = [w2v_model.wv.key_to_index[word] for word in document if word in w2v_model.wv.key_to_index]
            input_ids.append(torch.LongTensor(word_indices))

        # Pad sequences to the same length
        padded_input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True)

        # Convert to numpy array
        input_ids_array = padded_input_ids.numpy()

        self.embedding_matrix = w2v_model.wv.vectors
        self.embeddings = input_ids_array
        self.vocab_size = len(w2v_model.wv.key_to_index)
        self.embed_dim = w2v_model.vector_size

    def glove_embeddings(self, preprocessed_documents):
        glove_model = api.load("glove-wiki-gigaword-50")
        glove_index_dict = {word: i for i, word in enumerate(glove_model.index_to_key)}

        document_embeddings = []
        input_ids = []
        for document in preprocessed_documents:
            word_indices = [glove_index_dict[word] for word in document if word in glove_index_dict]
            input_ids.append(torch.LongTensor(word_indices))

            word_embeddings = [glove_model[word] for word in document if word in glove_model]
            if word_embeddings:
                document_embedding = np.mean(word_embeddings, axis=0)  # Average the word embeddings
            else:
                document_embedding = np.zeros(glove_model.vector_size)  # Use zero vector if no embeddings found
            document_embeddings.append(document_embedding)

        # Pad sequences to the same length
        padded_input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True)

        # Convert to numpy array
        input_ids_array = padded_input_ids.numpy()

        self.embedding_matrix = glove_model.vectors
        self.embeddings = input_ids_array
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

        input_ids = []
        for document_embedding in document_embeddings:
            word_indices = [fasttext_model.wv.key_to_index[word] for word in document if
                            word in fasttext_model.wv.key_to_index]
            input_ids.append(torch.LongTensor(word_indices))

        # Pad sequences to the same length
        padded_input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True)

        # Convert to numpy array
        input_ids_array = padded_input_ids.numpy()

        self.embedding_matrix = fasttext_model.wv.vectors
        self.embeddings = input_ids_array
        self.vocab_size = len(fasttext_model.wv.key_to_index)
        self.embed_dim = fasttext_model.vector_size

    def generate_bert_inputs(self, document):
        tokens = self.tokenizer.tokenize(document)

        # Set the maximum window size and stride
        window_size = 512
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
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')

        input_id_list = []
        attention_mask_list = []

        for document in preprocessed_documents:
            input_ids, attention_mask = self.generate_bert_inputs(document)
            input_id_list.append(input_ids)
            attention_mask_list.append(attention_mask)

        # labels = torch.tensor(labels, dtype=torch.long)
        input_ids_list = torch.stack(input_id_list, dim=0)
        attention_mask_list = torch.stack(attention_mask_list, dim=0)

        with torch.no_grad():
            outputs = model(input_ids_list, attention_mask=attention_mask_list)

        self.embeddings = outputs.last_hidden_state

        self.vocab_size = self.tokenizer.vocab_size
        self.embed_dim = self.embeddings.shape[2]

        self.embedding_matrix = model.embeddings.word_embeddings.weight
        
        self.embeddings = self.embeddings.view(self.embeddings.size(0), -1)
