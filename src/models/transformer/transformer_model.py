import os
import warnings

import numpy as np
import sklearn.exceptions
import torch
from scipy import interp
from sklearn.metrics import auc
from sklearn.model_selection import StratifiedKFold
from transformers import *

from src.config.model_args import ModelArgs
from src.models.metrics_history import MetricsHistory, calculate_metrics, calculate_classification_report
from src.models.model import Model
from src.models.transformer.transformer_data_module import TransformerDataModule

warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
try:
    import wandb

    wandb_available = True
except ImportError:
    wandb_available = False


class TransformerModel(Model):
    def train(self, config):
        MODEL_CLASSES = {
            'bert-base-cased': (BertConfig, BertForSequenceClassification, BertTokenizerFast),
            "allenai/longformer-base-4096": (
                LongformerConfig, LongformerForSequenceClassification, LongformerTokenizerFast),
            "roberta-base": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizerFast),
            "xlm-mlm-en-2048": (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
            "xlm-roberta-base": (XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizerFast),
            "xlnet-base-cased": (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizerFast)
        }

        self.args = ModelArgs()

        self.config_class, self.model_class, self.tokenizer_class = MODEL_CLASSES[self.model_name]

        if self.tokenizer_name is None:
            self.tokenizer_name = self.model_name

        self.data_module = TransformerDataModule(self.dataset, self.dataset_dir, config['batch_size'],
                                                 self.training_strategy,
                                                 config['apply_balance'], None, self.tokenizer_name,
                                                 config['do_lower_case'],
                                                 self.tokenizer_class)

        self.label_list = self.data_module.get_label_list()
        self.pos_label = self.data_module.get_pos_label()
        self.pos_label = self.pos_label if self.pos_label > 1 else None

        self.num_labels = len(self.label_list)
        len_labels_list = 2 if not self.num_labels else self.num_labels
        self.args.labels_list = [i for i in range(len_labels_list)]

        self.model_config = self.config_class.from_pretrained(self.model_name, num_labels=self.num_labels)

        self.model = self.model_class.from_pretrained(self.model_name, config=self.model_config)
        self.model.to(self.device)

        model_vars = [i for i in self.model.parameters()]
        self.optimizer = torch.optim.AdamW(model_vars, lr=config['learning_rate'])

        if torch.cuda.is_available():
            self.model.cuda()
            if config['multi_gpu']:
                self.model = torch.nn.DataParallel(self.model)

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
        avg_train_accuracy = 0.0
        avg_train_f1 = 0.0
        avg_train_precision = 0.0
        avg_train_recall = 0.0
        mean_auc_train = 0.0
        avg_val_accuracy = 0.0
        avg_val_f1 = 0.0
        avg_val_precision = 0.0
        avg_val_recall = 0.0
        mean_auc_val = 0.0

        # If zero-shot testing, we pass whole dataset to test_dataset
        if self.training_strategy != 'best_modal_on_wandb_RQ4':
            for train_index, val_index in kf.split(inputs, labels):
                history, tprs_train, aucs_train, tprs_val, aucs_val = self.train_transformer(history, train_index,
                                                                                             val_index,
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

        history, tprs_test, aucs_test = self.test_transformer(history, tprs_test, mean_fpr_test, aucs_test)

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

    def test_transformer(self, history, tprs_test, mean_fpr_test, aucs_test):
        self.test_dataloader = self.data_module.get_test_dataloader()

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

    def train_transformer(self, history, train_index, val_index, config, tprs_train, mean_fpr_train, aucs_train,
                          tprs_val,
                          mean_fpr_val, aucs_val):
        self.train_dataloader, self.val_dataloader = self.data_module.get_data_loaders(train_index, val_index)

        if config['apply_scheduler']:
            num_train_examples = len(self.train_dataloader)
            num_train_steps = int(num_train_examples / config['batch_size'] * config['num_train_epochs'])
            num_warmup_steps = int(num_train_steps * config['warmup_proportion'])

            scheduler = get_constant_schedule_with_warmup(self.optimizer, num_warmup_steps=num_warmup_steps)

        for epoch_i in range(config['num_train_epochs']):
            tr_loss = 0
            train_accuracy = []
            train_f1 = []
            train_precision = []
            train_recall = []

            self.model.train()

            for step, batch in enumerate(self.train_dataloader):
                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)

                outputs = self.model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
                loss, logits = outputs[:2]

                b_labels = b_labels.cpu().numpy()
                preds = torch.argmax(logits, dim=1).flatten()
                preds = preds.cpu().numpy()

                metrics = calculate_metrics(b_labels, preds, self.pos_label)

                tprs_train.append(interp(mean_fpr_train, metrics['fpr'], metrics['tpr']))
                roc_auc = auc(metrics['fpr'], metrics['tpr'])
                aucs_train.append(roc_auc)

                train_accuracy.append(metrics['accuracy'])
                train_f1.append(metrics['f1'])
                train_precision.append(metrics['precision'])
                train_recall.append(metrics['recall'])

                loss.backward()
                tr_loss += loss.item()
                self.optimizer.step()
                if config['apply_scheduler']:
                    scheduler.step()  # Update learning rate schedule
                self.model.zero_grad()

            avg_train_loss = tr_loss / len(self.train_dataloader)

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

    def test(self, mode):
        self.model.eval()

        total_test_loss = 0
        all_labels_ids = []

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

        nll_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)

        dataloader = self.val_dataloader

        if mode == 'test':
            dataloader = self.test_dataloader

        for batch in dataloader:
            b_input_ids = batch[0].to(self.device)
            b_input_mask = batch[1].to(self.device)
            b_labels = batch[2].to(self.device)

            with torch.no_grad():
                model_outputs = self.model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)

                loss, logits = model_outputs[:2]

                multi_label = len(self.label_list) > 2
                if multi_label:
                    logits = logits.sigmoid()

                total_test_loss += nll_loss(logits, b_labels)

            preds = logits.argmax(dim=1).cpu().numpy()
            all_labels_ids += b_labels.detach().cpu()

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
