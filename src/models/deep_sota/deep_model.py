import os
import warnings

import numpy as np
import sklearn.exceptions
import torch
import torch.optim as optim
from scipy import interp
from sklearn.metrics import auc
from sklearn.model_selection import StratifiedKFold

from src.models.deep_sota.bigram_cnn import BigramCNN
from src.models.deep_sota.cnn import CNN
from src.models.deep_sota.deep_data_module import DeepDataModule
from src.models.deep_sota.lstm import LstmClassification, BiLSTM
from src.models.metrics_history import MetricsHistory, calculate_metrics, calculate_classification_report
from src.models.model import Model

warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
try:
    import wandb

    wandb_available = True
except ImportError:
    wandb_available = False


class DeepModel(Model):
    def train(self, config):
        MODEL_CLASSES = {
            'cnn_ngram': BigramCNN,
            'lstm': LstmClassification,
            'cnn': CNN,
            'bilstm': BiLSTM
        }

        self.model_class = MODEL_CLASSES[self.model_name]
        self.data_module = DeepDataModule(self.dataset, self.dataset_dir, config['batch_size'], self.training_strategy,
                                          embedding_model_name=self.embedding_model_name,
                                          embedding_class=self.embedding_class)

        self.label_list = self.data_module.get_label_list()
        self.pos_label = self.data_module.get_pos_label()
        self.pos_label = self.pos_label if self.pos_label > 1 else None

        inputs, labels = self.data_module.get_data_all()

        vocab_size, embed_dim, self.embeddings = self.data_module.get_embeddings()

        self.initialize_model(config,
                              vocab_size=vocab_size,
                              num_classes=len(self.label_list),
                              embed_dim=embed_dim,
                              learning_rate=config['learning_rate'],
                              dropout=config['out_dropout_rate'],
                              batch_size=config['batch_size'])

        history = MetricsHistory()

        # TODO: select kfold iter from param
        kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

        tprs_val = []
        aucs_val = []
        mean_fpr_val = np.linspace(0, 1, 100)
        tprs_test = []
        aucs_test = []
        mean_fpr_test = np.linspace(0, 1, 100)
        tprs_train = []
        aucs_train = []
        mean_fpr_train = np.linspace(0, 1, 100)

        for train_index, val_index in kf.split(inputs, labels):
            history, tprs_train, aucs_train, tprs_val, aucs_val = self.train_DL(history, train_index, val_index,
                                                                                config, tprs_train,
                                                                                mean_fpr_train,
                                                                                aucs_train, tprs_val,
                                                                                mean_fpr_val,
                                                                                aucs_val)

        avg_train_accuracy, avg_train_f1, avg_train_precision, avg_train_recall, _ = history.calculate_average(
            history.train_metrics)
        avg_val_accuracy, avg_val_f1, avg_val_precision, avg_val_recall, _ = history.calculate_average(
            history.val_metrics)

        mean_tpr_train = np.mean(tprs_train, axis=0)
        mean_auc_train = auc(mean_fpr_train, mean_tpr_train)

        mean_tpr_val = np.mean(tprs_val, axis=0)
        mean_auc_val = auc(mean_fpr_val, mean_tpr_val)

        history, tprs_test, aucs_test = self.test_DL(history, tprs_test, mean_fpr_test, aucs_test)

        avg_test_accuracy, avg_test_f1, avg_test_precision, avg_test_recall, _ = history.calculate_average(
            history.test_metrics)
        _, avg_test_1_f1, avg_test_1_precision, avg_test_1_recall, avg_test_1_support = history.calculate_average(
            history.test_1_metrics)
        _, avg_test_0_f1, avg_test_0_precision, avg_test_0_recall, avg_test_0_support = history.calculate_average(
            history.test_0_metrics)

        mean_tpr_test = np.mean(tprs_test, axis=0)
        mean_auc_test = auc(mean_fpr_test, mean_tpr_test)

        if self.is_Wandb:
            wandb.log({
                "train_accuracy": avg_train_accuracy,
                "train_f1": avg_train_f1,
                "train_precision": avg_train_precision,
                "train_recall": avg_train_recall,
                'train_auc': mean_auc_train,

                "test_accuracy": avg_test_accuracy,
                "test_f1": avg_test_f1,
                "test_precision": avg_test_precision,
                "test_recall": avg_test_recall,
                'test_auc': mean_auc_test,

                "test_1_f1": avg_test_1_f1,
                "test_1_precision": avg_test_1_precision,
                "test_1_recall": avg_test_1_recall,
                'test_1_support': avg_test_1_support,

                "test_0_f1": avg_test_0_f1,
                "test_0_precision": avg_test_0_precision,
                "test_0_recall": avg_test_0_recall,
                'test_0_support': avg_test_0_support,

                "val_accuracy": avg_val_accuracy,
                "val_f1": avg_val_f1,
                "val_precision": avg_val_precision,
                "val_recall": avg_val_recall,
                'val_auc': mean_auc_val
            })

        if self.save_model:
            os.makedirs(self.output_dir, exist_ok=True)
            torch.save(self.model.state_dict(), 'outputs/model.pth')

    def train_DL(self, history, train_index, val_index, config, tprs_train, mean_fpr_train, aucs_train, tprs_val,
                 mean_fpr_val, aucs_val):
        self.train_dataloader, self.val_dataloader = self.data_module.get_data_loaders(train_index, val_index)

        for epoch_i in range(config['num_train_epochs']):
            total_loss = 0
            train_accuracy = []
            train_f1 = []
            train_precision = []
            train_recall = []

            self.model.train()

            # add mini batch here
            for train_input, train_label in self.train_dataloader:
                b_input_ids, b_labels = train_input.to(self.device), train_label.to(self.device)

                self.model.zero_grad()

                logits = self.model(b_input_ids)
                loss = self.loss_fn(logits, b_labels)

                total_loss += loss.item()

                preds = torch.argmax(logits, dim=1).flatten()
                preds = preds.cpu().numpy()

                b_labels = b_labels.cpu().numpy()

                metrics = calculate_metrics(b_labels, preds, self.pos_label)

                tprs_train.append(interp(mean_fpr_train, metrics['fpr'], metrics['tpr']))
                roc_auc = auc(metrics['fpr'], metrics['tpr'])
                aucs_train.append(roc_auc)

                train_accuracy.append(metrics['accuracy'])
                train_f1.append(metrics['f1'])
                train_precision.append(metrics['precision'])
                train_recall.append(metrics['recall'])

                loss.backward()
                self.optimizer.step()

            avg_train_loss = total_loss / len(self.train_dataloader)

            epoch_training_accuracy = np.mean(train_accuracy)
            epoch_training_f1 = np.mean(train_f1)
            epoch_training_precision = np.mean(train_precision)
            epoch_training_recall = np.mean(train_recall)

            history.add_train_metrics(
                {'accuracy': epoch_training_accuracy,
                 'f1': epoch_training_f1,
                 'precision': epoch_training_precision,
                 'recall': epoch_training_recall
                 }
            )

            if self.val_dataloader is not None:
                val_metrics = self.test('val')

                val_fpr = [list(item) for item in val_metrics['fpr']][0]
                val_tpr = [list(item) for item in val_metrics['tpr']][0]
                tprs_val.append(interp(mean_fpr_val, val_fpr, val_tpr))
                roc_auc = auc(val_fpr, val_tpr)
                aucs_val.append(roc_auc)

                history.add_val_metrics(val_metrics['test_metrics'])

        return history, tprs_train, aucs_train, tprs_val, aucs_val

    def test_DL(self, history, tprs_test, mean_fpr_test, aucs_test):
        self.test_dataloader = self.data_module.get_test_dataloader()

        if self.model_name == 'gan-bert':
            test_metrics = self.test_gan_bert('test')
        else:
            test_metrics = self.test('test')

        test_fpr = [list(item) for item in test_metrics['fpr']][0]
        test_tpr = [list(item) for item in test_metrics['tpr']][0]
        tprs_test.append(interp(mean_fpr_test, test_fpr, test_tpr))
        roc_auc = auc(test_fpr, test_tpr)
        aucs_test.append(roc_auc)

        history.add_test_metrics(test_metrics['test_metrics'])
        history.add_test_0_metrics(test_metrics['test_0_metrics'])
        history.add_test_1_metrics(test_metrics['test_1_metrics'])

        return history, tprs_test, aucs_test

    def test(self, mode):
        self.model.eval()

        accuracy = []
        test_loss = []
        recall = []
        f1 = []
        precision = []
        fpr = []
        tpr = []
        thresholds = []

        test_0_support = []
        test_0_recall = []
        test_0_f1 = []
        test_0_precision = []

        test_1_support = []
        test_1_recall = []
        test_1_f1 = []
        test_1_precision = []

        dataloader = self.val_dataloader
        if mode == 'test':
            dataloader = self.test_dataloader

        for inputs, labels in dataloader:
            b_input_ids, b_labels = inputs.to(self.device), labels.to(self.device)

            with torch.no_grad():
                logits = self.model(b_input_ids)

            loss = self.loss_fn(logits, b_labels)
            test_loss.append(loss.item())

            preds = torch.argmax(logits, dim=1).flatten()
            preds = preds.cpu().numpy()

            b_labels = b_labels.cpu().numpy()

            class_index = len(self.label_list) - 1
            class_index = str(class_index)

            test_metrics = calculate_classification_report(b_labels, preds, self.pos_label, class_index)

            test_1_support.append(test_metrics['test_1_metrics']['support'])
            test_1_recall.append(test_metrics['test_1_metrics']['recall'])
            test_1_f1.append(test_metrics['test_1_metrics']['f1'])
            test_1_precision.append(test_metrics['test_1_metrics']['precision'])

            test_0_support.append(test_metrics['test_0_metrics']['support'])
            test_0_recall.append(test_metrics['test_0_metrics']['recall'])
            test_0_f1.append(test_metrics['test_0_metrics']['f1'])
            test_0_precision.append(test_metrics['test_0_metrics']['precision'])

            accuracy.append(test_metrics['test_metrics']['accuracy'])
            recall.append(test_metrics['test_metrics']['recall'])
            f1.append(test_metrics['test_metrics']['f1'])
            precision.append(test_metrics['test_metrics']['precision'])

            fpr.append(test_metrics['fpr'])
            tpr.append(test_metrics['tpr'])
            thresholds.append(test_metrics['thresholds'])

        return {
            'loss': np.mean(test_loss),
            'test_metrics': {
                'accuracy': np.mean(accuracy),
                'f1': np.mean(f1),
                'precision': np.mean(precision),
                'recall': np.mean(recall)
            },
            'test_0_metrics': {
                'support': np.mean(test_0_support),
                'recall': np.mean(test_0_recall),
                'precision': np.mean(test_0_precision),
                'f1': np.mean(test_0_f1)
            },
            'test_1_metrics': {
                'support': np.mean(test_1_support),
                'recall': np.mean(test_1_recall),
                'precision': np.mean(test_1_precision),
                'f1': np.mean(test_1_f1)
            },
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds}

    def initialize_model(self, config, pretrained_embedding=None,
                         freeze_embedding=False,
                         vocab_size=None,
                         embed_dim=300,
                         filter_sizes=[3, 4, 5],
                         num_filters=[100, 100, 100],
                         num_classes=2,
                         dropout=0.5,
                         learning_rate=0.01, batch_size=4):

        # TODO: these are redundant with model class selection, remove to init class of each model
        if self.model_name == 'cnn':
            # TODO: do with static and non-static cnn
            # TODO: add char and word n-grams
            self.model = CNN(pretrained_embedding=self.embeddings,
                             freeze_embedding=freeze_embedding,
                             vocab_size=vocab_size,
                             embed_dim=embed_dim,
                             filter_sizes=filter_sizes,
                             num_filters=num_filters,
                             num_classes=num_classes,
                             dropout=dropout,
                             embedding_class=self.embedding_class)
        elif self.model_name == 'cnn_ngram':
            self.model = BigramCNN(embed_dim, self.embeddings, num_classes, num_filters=100, window_size=2,
                                   embedding_class=self.embedding_class)
        elif self.model_name == 'bilstm':
            self.model = BiLSTM(embed_size=embed_dim, embedding=self.embeddings,
                                out_features=num_classes, batch_size=batch_size, embedding_class=self.embedding_class)
        else:
            self.model = LstmClassification(embed_size=embed_dim, embedding=self.embeddings,
                                            out_features=num_classes, batch_size=batch_size,
                                            embedding_class=self.embedding_class)
        self.model.to(self.device)
        self.optimizer = optim.Adadelta(self.model.parameters(),
                                        lr=learning_rate,
                                        rho=0.95)  # TODO: magic num
