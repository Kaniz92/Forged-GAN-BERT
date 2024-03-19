import heapq
from collections import Counter

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
nltk.download('stopwords')
nltk.download('punkt')
from sklearn.linear_model import LogisticRegression
from datasets import load_dataset

# BERT classifier
# Installing a custom version of Simple Transformers
# !git clone https://github.com/NVIDIA/apex
# !pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex
# !pip install --upgrade tqdm
# !pip install transformers
# !pip install tensorboardX
# !pip install simpletransformers

from simpletransformers.classification import ClassificationModel
from pandarallel import pandarallel
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


class BaselineModule:
    def __init__(
            self,
            dataset_name,
            dataset_dir,
            is_local = False,
            source='enron',
            recompute=False
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir
        self.is_local = is_local
        self.source = source
        self.recompute = recompute
        self.stop_words = set(stopwords.words('english'))
        self.ps = PorterStemmer()

        pandarallel.initialize()

    def calculate_metrics(self):
        print("Loading and processing dataframe")
        list_senders = [2]

        feature_list = ["avg_len", "len_text", "len_words", "num_short_w", "per_digit", "per_cap", "f_a", "f_b",
                        "f_c", "f_d",
                        "f_e", "f_f", "f_g", "f_h", "f_i", "f_j", "f_k", "f_l", "f_m", "f_n", "f_o", "f_p", "f_q",
                        "f_r",
                        "f_s",
                        "f_t", "f_u", "f_v", "f_w", "f_x", "f_y", "f_z", "f_0", "f_1", "f_2", "f_3", "f_4", "f_5",
                        "f_6",
                        "f_7",
                        "f_8", "f_9", "f_e_0", "f_e_1", "f_e_2", "f_e_3", "f_e_4", "f_e_5", "f_e_6", "f_e_7",
                        "f_e_8", "f_e_9",
                        "f_e_10", "f_e_11", "richness"]

        train_df = pd.DataFrame(
            load_dataset(f"Authorship/{self.dataset_dir}", data_files=[f"{self.dataset_name}_train_val.csv"])['train'])
        nlp_train, ind_train = self.update_df(train_df, feature_list)

        test_df = pd.DataFrame(
            load_dataset(f"Authorship/{self.dataset_dir}", data_files=[f"{self.dataset_name}_test.csv"])['train'])
        nlp_test, ind_test = self.update_df(test_df, feature_list)

        sub_df = pd.concat([nlp_train, nlp_test], ignore_index=True)
        text = " ".join(sub_df['content'].values)
        list_bigram = self.return_best_bi_grams(text)
        list_trigram = self.return_best_tri_grams(text)

        list_scores = []
        list_f1 = []

        for limit in list_senders:
            print("Number of speakers : ", limit)
            # Bert + Classification Layer
            print("#####")
            print("Training BERT")

            model = ClassificationModel('bert', 'bert-base-cased', num_labels=limit,
                                        args={'reprocess_input_data': True, 'overwrite_output_dir': True,
                                              'num_train_epochs': 5, 'use_multiprocessing': False,
                                              'use_multiprocessing_for_evaluation': False}, use_cuda=not self.is_local)
            model.train_model(nlp_train[['content', 'Target']])

            predictions, raw_outputs = model.predict(list(nlp_test['content']))
            score_bert = accuracy_score(predictions, nlp_test['Target'])
            f1_bert = f1_score(predictions, nlp_test['Target'], average="macro")

            predictions, raw_out_train = model.predict(list(nlp_train['content']))

            print("Training done, accuracy is : ", score_bert)
            print("Training done, f1-score is : ", f1_bert)

            # Style-based classifier
            print("#####")
            print("Training style classifier")

            X_style_train = nlp_train[feature_list]
            X_style_test = nlp_test[feature_list]

            clf = LogisticRegression(random_state=0).fit(X_style_train, nlp_train['Target'])
            y_pred = clf.predict(X_style_test)
            y_proba = clf.predict_proba(X_style_test)
            y_proba_train = clf.predict_proba(X_style_train)
            score_style = accuracy_score(nlp_test['Target'], y_pred)
            f1_style = f1_score(nlp_test['Target'], y_pred, average="macro")

            print("Training done, accuracy is : ", score_style)
            print("Training done, f1-score is : ", f1_style)

            # Model Combination
            print("#####")
            print("Model combination")

            feat_for_BERT_LR_train = np.concatenate([raw_out_train, y_proba_train], axis=1)
            feat_for_BERT_LR_test = np.concatenate([raw_outputs, y_proba], axis=1)

            clf = LogisticRegression(random_state=0).fit(feat_for_BERT_LR_train, nlp_train['Target'])
            y_pred = clf.predict(feat_for_BERT_LR_test)
            score_comb = accuracy_score(nlp_test['Target'], y_pred)
            f1_comb = f1_score(nlp_test['Target'], y_pred, average="macro")

            print("Training done, accuracy is : ", score_comb)
            print("Training done, f1-score is : ", f1_comb)

            # Character N-gram only
            print("#####")
            print("Character N-gram")

            feats_train = nlp_train['content'].apply(
                lambda x: self.find_freq_n_gram_in_txt(x, list_bigram, list_trigram)).values
            feats_test = nlp_test['content'].apply(
                lambda x: self.find_freq_n_gram_in_txt(x, list_bigram, list_trigram)).values

            feats_train = pd.DataFrame(feats_train)[0].apply(lambda x: pd.Series(x))
            feats_test = pd.DataFrame(feats_test)[0].apply(lambda x: pd.Series(x))

            clf_char = LogisticRegression(random_state=0).fit(feats_train, nlp_train['Target'])
            y_pred = clf_char.predict(feats_test)
            y_proba = clf_char.predict_proba(feats_test)
            y_proba_train = clf_char.predict_proba(feats_train)

            score_char = accuracy_score(nlp_test['Target'], y_pred)
            f1_char = f1_score(nlp_test['Target'], y_pred, average="macro")

            print("Training done, accuracy is : ", score_char)
            print("Training done, f1-score is : ", f1_char)

            # BERT + Style + Char N-gram
            print("#####")
            print("BERT + Style + Char N-gram")

            feat_for_BERT_full_train = np.concatenate([feat_for_BERT_LR_train, y_proba_train], axis=1)
            feat_for_BERT_full_test = np.concatenate([feat_for_BERT_LR_test, y_proba], axis=1)

            clf = LogisticRegression(random_state=0).fit(feat_for_BERT_full_train, nlp_train['Target'])

            y_pred = clf.predict(feat_for_BERT_full_test)
            score_comb_fin = accuracy_score(nlp_test['Target'], y_pred)
            f1_comb_fin = f1_score(nlp_test['Target'], y_pred, average="macro")
            print("Training done, accuracy is : ", score_comb_fin)
            print("Training done, f1-score is : ", f1_comb_fin)

            # Store scores
            list_scores.append([limit, score_bert, score_style, score_comb, score_char, score_comb_fin])
            list_f1.append([limit, f1_bert, f1_style, f1_comb, f1_char, f1_comb_fin])

        list_scores = np.array(list_scores)

        return list_scores, list_f1

    def max_content_length(self, bookText):
        result = [i.split()[:512] for i in bookText]
        return result
    def update_df(self, df, feature_list):
        df['From'] = df['AuthorID']
        df['BookText'] = df['BookText'].apply(lambda x: x.ljust(512)[:512])
        # df['BookText'] = df['BookText'].apply(lambda x: self.max_content_length(str(x)))
        df['content_tfidf'] = df['BookText'].parallel_apply(lambda x: self.process(x))
        df[feature_list] = df['BookText'].parallel_apply(lambda x: self.extract_style(x))
        df['content'] = df['BookText']

        new_feature_list = ['From', 'content', 'content_tfidf'] + feature_list
        df = df[new_feature_list]
        df = df.dropna()

        dict_nlp_enron = {}
        k = 0

        for val in np.unique(df.From):
            dict_nlp_enron[val] = k
            k += 1

        df['Target'] = df['From'].apply(lambda x: dict_nlp_enron[x])
        ind_train = list(df.index)

        return df, ind_train

    def fil_sent(self, sent):
        filtered_sentence = ' '.join([w for w in sent.split() if not w in self.stop_words])
        return filtered_sentence

    def process(self, sent):
        sent = str(sent)
        return self.fil_sent(' '.join([self.ps.stem(str(x).lower()) for x in word_tokenize(sent)]))

    def extract_style(self, text):
        text = str(text)
        len_text = len(text)
        len_words = len(text.split())
        avg_len = np.mean([len(t) for t in text.split()])
        num_short_w = len([t for t in text.split() if len(t) < 3])
        per_digit = sum(t.isdigit() for t in text) / len(text)
        per_cap = sum(1 for t in text if t.isupper()) / len(text)
        f_a = sum(1 for t in text if t.lower() == "a") / len(text)
        f_b = sum(1 for t in text if t.lower() == "b") / len(text)
        f_c = sum(1 for t in text if t.lower() == "c") / len(text)
        f_d = sum(1 for t in text if t.lower() == "d") / len(text)
        f_e = sum(1 for t in text if t.lower() == "e") / len(text)
        f_f = sum(1 for t in text if t.lower() == "f") / len(text)
        f_g = sum(1 for t in text if t.lower() == "g") / len(text)
        f_h = sum(1 for t in text if t.lower() == "h") / len(text)
        f_i = sum(1 for t in text if t.lower() == "i") / len(text)
        f_j = sum(1 for t in text if t.lower() == "j") / len(text)
        f_k = sum(1 for t in text if t.lower() == "k") / len(text)
        f_l = sum(1 for t in text if t.lower() == "l") / len(text)
        f_m = sum(1 for t in text if t.lower() == "m") / len(text)
        f_n = sum(1 for t in text if t.lower() == "n") / len(text)
        f_o = sum(1 for t in text if t.lower() == "o") / len(text)
        f_p = sum(1 for t in text if t.lower() == "p") / len(text)
        f_q = sum(1 for t in text if t.lower() == "q") / len(text)
        f_r = sum(1 for t in text if t.lower() == "r") / len(text)
        f_s = sum(1 for t in text if t.lower() == "s") / len(text)
        f_t = sum(1 for t in text if t.lower() == "t") / len(text)
        f_u = sum(1 for t in text if t.lower() == "u") / len(text)
        f_v = sum(1 for t in text if t.lower() == "v") / len(text)
        f_w = sum(1 for t in text if t.lower() == "w") / len(text)
        f_x = sum(1 for t in text if t.lower() == "x") / len(text)
        f_y = sum(1 for t in text if t.lower() == "y") / len(text)
        f_z = sum(1 for t in text if t.lower() == "z") / len(text)
        f_1 = sum(1 for t in text if t.lower() == "1") / len(text)
        f_2 = sum(1 for t in text if t.lower() == "2") / len(text)
        f_3 = sum(1 for t in text if t.lower() == "3") / len(text)
        f_4 = sum(1 for t in text if t.lower() == "4") / len(text)
        f_5 = sum(1 for t in text if t.lower() == "5") / len(text)
        f_6 = sum(1 for t in text if t.lower() == "6") / len(text)
        f_7 = sum(1 for t in text if t.lower() == "7") / len(text)
        f_8 = sum(1 for t in text if t.lower() == "8") / len(text)
        f_9 = sum(1 for t in text if t.lower() == "9") / len(text)
        f_0 = sum(1 for t in text if t.lower() == "0") / len(text)
        f_e_0 = sum(1 for t in text if t.lower() == "!") / len(text)
        f_e_1 = sum(1 for t in text if t.lower() == "-") / len(text)
        f_e_2 = sum(1 for t in text if t.lower() == ":") / len(text)
        f_e_3 = sum(1 for t in text if t.lower() == "?") / len(text)
        f_e_4 = sum(1 for t in text if t.lower() == ".") / len(text)
        f_e_5 = sum(1 for t in text if t.lower() == ",") / len(text)
        f_e_6 = sum(1 for t in text if t.lower() == ";") / len(text)
        f_e_7 = sum(1 for t in text if t.lower() == "'") / len(text)
        f_e_8 = sum(1 for t in text if t.lower() == "/") / len(text)
        f_e_9 = sum(1 for t in text if t.lower() == "(") / len(text)
        f_e_10 = sum(1 for t in text if t.lower() == ")") / len(text)
        f_e_11 = sum(1 for t in text if t.lower() == "&") / len(text)
        richness = len(list(set(text.split()))) / len(text.split())

        return pd.Series(
            [avg_len, len_text, len_words, num_short_w, per_digit, per_cap, f_a, f_b, f_c, f_d, f_e, f_f, f_g, f_h, f_i,
             f_j, f_k, f_l, f_m, f_n, f_o, f_p, f_q, f_r, f_s, f_t, f_u, f_v, f_w, f_x, f_y, f_z, f_0, f_1, f_2, f_3,
             f_4, f_5, f_6, f_7, f_8, f_9, f_e_0, f_e_1, f_e_2, f_e_3, f_e_4, f_e_5, f_e_6, f_e_7, f_e_8, f_e_9, f_e_10,
             f_e_11, richness])

    def return_best_bi_grams(self, text):
        bigrams = ngrams(text, 2)

        data = dict(Counter(bigrams))
        list_ngrams = heapq.nlargest(100, data.keys(), key=lambda k: data[k])
        return list_ngrams

    def return_best_tri_grams(self, text):
        trigrams = ngrams(text, 3)

        data = dict(Counter(trigrams))
        list_ngrams = heapq.nlargest(100, data.keys(), key=lambda k: data[k])
        return list_ngrams

    def find_freq_n_gram_in_txt(self, text, list_bigram, list_trigram):

        to_ret = []

        num_bigrams = len(Counter(zip(text, text[1:])))
        num_trigrams = len(Counter(zip(text, text[1:], text[2:])))

        for n_gram in list_bigram:
            to_ret.append(text.count(''.join(n_gram)) / num_bigrams)

        for n_gram in list_trigram:
            to_ret.append(text.count(''.join(n_gram)) / num_trigrams)

        return to_ret
