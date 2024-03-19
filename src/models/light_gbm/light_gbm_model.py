import os
import warnings

import lightgbm as lgb
import numpy as np
import sklearn.exceptions
import torch
from scipy import interp
from sklearn.metrics import auc
from sklearn.model_selection import StratifiedKFold

from src.models.light_gbm.light_gbm_data_module import LightGBMDataModule
from src.models.metrics_history import MetricsHistory, calculate_metrics, calculate_classification_report
from src.models.model import Model

warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
try:
    import wandb

    wandb_available = True
except ImportError:
    wandb_available = False


class LightGBMModel(Model):
    def train(self, config):
        # TODO: move to config
        self.params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': 'binary_logloss',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 0
        }

        self.data_module = LightGBMDataModule(self.dataset, self.dataset_dir, config['batch_size'],
                                              self.training_strategy,
                                              embedding_model_name=self.embedding_model_name,
                                              embedding_class=self.embedding_class)

        self.label_list = self.data_module.get_label_list()
        self.pos_label = self.data_module.get_pos_label()
        self.pos_label = self.pos_label if self.pos_label > 1 else None

        inputs, labels = self.data_module.get_data_all()

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
            history, tprs_train, aucs_train, tprs_val, aucs_val = self.train_lgb(history, inputs, labels,
                                                                                 train_index, val_index,
                                                                                 tprs_train, mean_fpr_train,
                                                                                 aucs_train, tprs_val, mean_fpr_val,
                                                                                 aucs_val)

        avg_train_accuracy, avg_train_f1, avg_train_precision, avg_train_recall, _ = history.calculate_average(
            history.train_metrics)
        avg_val_accuracy, avg_val_f1, avg_val_precision, avg_val_recall, _ = history.calculate_average(
            history.val_metrics)

        mean_tpr_train = np.mean(tprs_train, axis=0)
        mean_auc_train = auc(mean_fpr_train, mean_tpr_train)

        mean_tpr_val = np.mean(tprs_val, axis=0)
        mean_auc_val = auc(mean_fpr_val, mean_tpr_val)

        history, tprs_test, aucs_test = self.test_lgb(history, tprs_test, mean_fpr_test,
                                                     aucs_test)

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

    def train_lgb(self, history, inputs, labels, train_index, val_index, tprs_train, mean_fpr_train,
                 aucs_train, tprs_val, mean_fpr_val, aucs_val):

        x_train, x_val = inputs[train_index], inputs[val_index]
        y_train, y_val = labels[train_index], labels[val_index]
        train_data = lgb.Dataset(x_train, label=y_train)

        self.model = lgb.train(self.params, train_data, num_boost_round=100)
        y_pred_train = self.model.predict(x_train)
        y_pred_val = self.model.predict(x_val)

        threshold = 0.5
        y_pred_train_binary = (y_pred_train > threshold).astype(int)
        y_pred_val_binary = (y_pred_val > threshold).astype(int)

        train_metrics = calculate_metrics(y_train, y_pred_train_binary, self.pos_label)
        history.add_train_metrics(train_metrics)

        val_metrics = calculate_metrics(y_val, y_pred_val_binary, self.pos_label)
        history.add_val_metrics(val_metrics)

        # TODO: include tprs in MetricsHistory() too
        tprs_train.append(interp(mean_fpr_train, train_metrics['fpr'], train_metrics['tpr']))
        roc_auc = auc(train_metrics['fpr'], train_metrics['tpr'])
        aucs_train.append(roc_auc)

        tprs_val.append(interp(mean_fpr_val, train_metrics['fpr'], train_metrics['tpr']))
        roc_auc = auc(train_metrics['fpr'], train_metrics['tpr'])
        aucs_val.append(roc_auc)

        return history, tprs_train, aucs_train, tprs_val, aucs_val

    def test_lgb(self, history, tprs_test, mean_fpr_test, aucs_test):
        x_test, y_test = self.data_module.get_test_data()

        y_pred_test = self.model.predict(x_test)
        threshold = 0.5
        y_pred_test = (y_pred_test > threshold).astype(int)

        class_index = len(self.label_list) - 1
        class_index = str(class_index)

        metrics = calculate_classification_report(y_test, y_pred_test, self.pos_label, class_index)

        history.add_test_metrics(metrics['test_metrics'])
        history.add_test_0_metrics(metrics['test_0_metrics'])
        history.add_test_1_metrics(metrics['test_1_metrics'])

        tprs_test.append(interp(mean_fpr_test, metrics['fpr'], metrics['tpr']))
        roc_auc = auc(metrics['fpr'], metrics['tpr'])
        aucs_test.append(roc_auc)

        return history, tprs_test, aucs_test
