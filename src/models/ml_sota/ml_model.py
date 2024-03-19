import os
import warnings

import numpy as np
import sklearn.exceptions
import torch
from scipy import interp
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

from src.models.metrics_history import MetricsHistory, calculate_metrics, calculate_classification_report
from src.models.ml_sota.ml_data_module import MLDataModule
from src.models.model import Model

warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
try:
    import wandb

    wandb_available = True
except ImportError:
    wandb_available = False


class MLModel(Model):
    def train(self, config):
        MODEL_CLASSES = {
            'knn': KNeighborsClassifier(n_neighbors=3),
            'svm': LinearSVC(C=10, dual=False, max_iter=1),
            'naive_bayes': MultinomialNB(alpha=1e+10), #, force_alpha=True),
            'random_forest': RandomForestClassifier(max_depth=2, random_state=42)
        }
        # parameters = {"n_neighbors": [2, 4, 6, 8]} #KNN
        # parameters = {"C": [0.001, 0.01, 0.1, 1, 10, 100, 1000]} #SVM
        # parameters = {"alpha": [0.01, 0.1, 0.5, 1]} #multinomial NM
        # parameters = {"n_estimators": [10, 20, 30, 40]} #random forest

        self.model_class = MODEL_CLASSES[self.model_name]
        self.data_module = MLDataModule(self.dataset, self.dataset_dir, config['batch_size'], self.training_strategy,
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
            history, tprs_train, aucs_train, tprs_val, aucs_val = self.train_ML(history, inputs, labels,
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

        history, tprs_test, aucs_test = self.test_ML(history, tprs_test, mean_fpr_test,
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

    def train_ML(self, history, inputs, labels, train_index, val_index, tprs_train, mean_fpr_train,
                 aucs_train, tprs_val, mean_fpr_val, aucs_val):

        x_train, x_val = inputs[train_index], inputs[val_index]
        y_train, y_val = labels[train_index], labels[val_index]

        y_pred_train, y_pred_val = self.do_classify(x_train, y_train, x_val)

        train_metrics = calculate_metrics(y_train, y_pred_train, self.pos_label)
        history.add_train_metrics(train_metrics)

        val_metrics = calculate_metrics(y_val, y_pred_val, self.pos_label)
        history.add_val_metrics(val_metrics)

        # TODO: include tprs in MetricsHistory() too
        tprs_train.append(interp(mean_fpr_train, train_metrics['fpr'], train_metrics['tpr']))
        roc_auc = auc(train_metrics['fpr'], train_metrics['tpr'])
        aucs_train.append(roc_auc)

        tprs_val.append(interp(mean_fpr_val, train_metrics['fpr'], train_metrics['tpr']))
        roc_auc = auc(train_metrics['fpr'], train_metrics['tpr'])
        aucs_val.append(roc_auc)

        return history, tprs_train, aucs_train, tprs_val, aucs_val

    def test_ML(self, history, tprs_test, mean_fpr_test, aucs_test):
        x_test, y_test = self.data_module.get_test_data()

        y_pred_test = self.model.predict(x_test)

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

    @staticmethod
    def cv_optimize(clf, parameters, X, y, n_jobs=1, n_folds=5, score_func=None):
        if score_func:
            gs = GridSearchCV(clf, param_grid=parameters, cv=n_folds, n_jobs=n_jobs, scoring=score_func)
        else:
            gs = GridSearchCV(clf, param_grid=parameters, n_jobs=n_jobs, cv=n_folds)
        gs.fit(X, y)
        print("Best Parameters:", gs.best_params_, gs.best_score_)
        best = gs.best_estimator_
        return best

    def do_classify(self, x_train, y_train, x_val, parameters=None, score_func=None,
                    n_jobs=5, inputs=None, labels=None):
        if parameters:
            self.model = self.cv_optimize(self.model_class, parameters, inputs, labels, n_jobs=n_jobs, n_folds=7,
                                          score_func=score_func)

        self.model = self.model_class.fit(x_train, y_train)

        y_pred_train = self.model.predict(x_train)
        y_pred_val = self.model.predict(x_val)

        return y_pred_train, y_pred_val
