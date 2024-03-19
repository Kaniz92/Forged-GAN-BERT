import os
import warnings

import numpy as np
import sklearn.exceptions
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import interp
from sklearn.metrics import auc
from sklearn.model_selection import StratifiedKFold
from transformers import AutoModel, AutoConfig
from transformers import get_constant_schedule_with_warmup

from src.models.ganbert.ganbert_data_module import GANBertDataModule
from src.models.metrics_history import MetricsHistory, calculate_metrics, calculate_classification_report
from src.models.model import Model

warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
try:
    import wandb

    wandb_available = True
except ImportError:
    wandb_available = False


class GANBertModel(Model):
    def train(self, config):
        self.data_module = GANBertDataModule(self.dataset, self.dataset_dir, config['batch_size'],
                                             self.training_strategy,
                                             embedding_model_name=self.embedding_model_name)

        self.label_list = self.data_module.get_label_list()
        self.pos_label = self.data_module.get_pos_label()
        self.pos_label = self.pos_label if self.pos_label > 1 else None

        inputs, labels = self.data_module.get_data_all()

        self.initialize_model(config)

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
            history, tprs_train, aucs_train, tprs_val, aucs_val = self.train_gan_bert(history, train_index,
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

    def test_DL(self, history, tprs_test, mean_fpr_test, aucs_test):
        self.test_dataloader = self.data_module.get_test_dataloader()

        test_metrics = self.test_gan_bert('test')

        test_fpr = [list(item) for item in test_metrics['fpr']][0]
        test_tpr = [list(item) for item in test_metrics['tpr']][0]
        tprs_test.append(interp(mean_fpr_test, test_fpr, test_tpr))
        roc_auc = auc(test_fpr, test_tpr)
        aucs_test.append(roc_auc)

        history.add_test_metrics(test_metrics['test_metrics'])
        history.add_test_0_metrics(test_metrics['test_0_metrics'])
        history.add_test_1_metrics(test_metrics['test_1_metrics'])

        return history, tprs_test, aucs_test

    def train_gan_bert(self, history, train_index, val_index, config, tprs_train, mean_fpr_train, aucs_train, tprs_val,
                       mean_fpr_val, aucs_val):
        self.train_dataloader, self.val_dataloader = self.data_module.get_data_loaders(train_index, val_index)

        if config['apply_scheduler']:
            num_train_examples = len(self.train_dataloader)
            num_train_steps = int(num_train_examples / config['batch_size'] * config['num_train_epochs'])
            num_warmup_steps = int(num_train_steps * config['warmup_proportion'])

            scheduler_d = get_constant_schedule_with_warmup(self.dis_optimizer,
                                                            num_warmup_steps=num_warmup_steps)
            scheduler_g = get_constant_schedule_with_warmup(self.gen_optimizer,
                                                            num_warmup_steps=num_warmup_steps)

        for epoch_i in range(config['num_train_epochs']):
            tr_g_loss = 0
            tr_d_loss = 0
            train_accuracy = []
            train_f1 = []
            train_precision = []
            train_recall = []

            self.transformer.train()
            self.generator.train()
            self.discriminator.train()

            for step, batch in enumerate(self.train_dataloader):
                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)
                b_label_mask = batch[3].to(self.device)

                real_input_size = b_input_ids.shape[0]
                print(f'real_input_size={real_input_size}')
                

                # total input size to discriminator is 2* real_input_size
                # controlling fake percentage from this whole amount
                # fake_batch_size [0, 0.5] since default setting in GAN-BERT is 50% of fake
                fake_batch_size = int(self.fake_percentage * 2 * real_input_size)
                
                model_outputs = self.transformer(b_input_ids, attention_mask=b_input_mask)
                hidden_states = model_outputs[-1]

                # since we need to keep the total input size to discriminator constant
                # missing fake data is filled by copying from real text
                additional_hidden_states_split = real_input_size - fake_batch_size
                real_input_size += additional_hidden_states_split
                
                additional_hidden_states = hidden_states[:additional_hidden_states_split]
                hidden_states = torch.cat([hidden_states, additional_hidden_states])

                additional_labels = b_labels[:additional_hidden_states_split]
                b_labels = torch.cat([b_labels, additional_labels])

                additiona_label_masks = b_label_mask[:additional_hidden_states_split]
                b_label_mask = torch.cat([b_label_mask, additiona_label_masks])

                noise = torch.zeros(fake_batch_size, config['noise_size'], device=self.device).uniform_(0, 1)
                gen_rep = self.generator(noise)

                disciminator_input = torch.cat([hidden_states, gen_rep], dim=0)
                print(f'discriminator input size = {len(disciminator_input)}')
                print(f'fake_batch_size={fake_batch_size}')
                print(f'real_input_size={real_input_size}')
                print(f'additional_hidden_states_split={additional_hidden_states_split}')

                features, logits, probs = self.discriminator(disciminator_input)

                features_list = torch.split(features, real_input_size)
                print(f'features={len(features)}')
                
                D_real_features = features_list[0]

                if fake_batch_size == 0:
                  logits_list = torch.split(logits, real_input_size)
                  D_real_logits = logits_list[0]
                  probs_list = torch.split(probs, real_input_size)
                  D_real_probs = probs_list[0]

                  g_feat_reg = torch.mean(torch.pow(torch.mean(D_real_features, dim=0), 2))
                  g_loss =  g_feat_reg

                else:
                  D_fake_features = features_list[1]

                  logits_list = torch.split(logits, real_input_size)
                  D_real_logits = logits_list[0]

                  probs_list = torch.split(probs, real_input_size)
                  D_real_probs = probs_list[0]
                  D_fake_probs = probs_list[1]
                  
                  g_loss_d = -1 * torch.mean(torch.log(1 - D_fake_probs[:, -1] + config['epsilon']))
                
                  g_feat_reg = torch.mean(
                      torch.pow(torch.mean(D_real_features, dim=0) - torch.mean(D_fake_features, dim=0), 2))
                
                  g_loss = g_loss_d + g_feat_reg

                logits = D_real_logits[:, 0:-1]
                log_probs = F.log_softmax(logits, dim=-1)

                label2one_hot = torch.nn.functional.one_hot(b_labels, len(self.label_list))
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

                per_example_loss = -torch.sum(label2one_hot * log_probs, dim=-1)
                per_example_loss = torch.masked_select(per_example_loss, b_label_mask.to(self.device))
                labeled_example_count = per_example_loss.type(torch.float32).numel()

                if labeled_example_count == 0:
                    D_L_Supervised = 0
                else:
                    D_L_Supervised = torch.div(torch.sum(per_example_loss.to(self.device)), labeled_example_count)

                D_L_unsupervised1U = -1 * torch.mean(torch.log(1 - D_real_probs[:, -1] + config['epsilon']))
                
                if fake_batch_size == 0:
                  d_loss = D_L_Supervised + D_L_unsupervised1U
                else:
                  D_L_unsupervised2U = -1 * torch.mean(torch.log(D_fake_probs[:, -1] + config['epsilon']))
                  d_loss = D_L_Supervised + D_L_unsupervised1U + D_L_unsupervised2U

                self.gen_optimizer.zero_grad()
                self.dis_optimizer.zero_grad()

                g_loss.backward(retain_graph=True)
                d_loss.backward()

                self.gen_optimizer.step()
                self.dis_optimizer.step()

                tr_g_loss += g_loss.item()
                tr_d_loss += d_loss.item()

                if config['apply_scheduler']:
                    scheduler_d.step()
                    scheduler_g.step()

            avg_train_loss_g = tr_g_loss / len(self.train_dataloader)
            avg_train_loss_d = tr_d_loss / len(self.train_dataloader)

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
                val_metrics = self.test_gan_bert('val')

                val_fpr = [list(item) for item in val_metrics['fpr']][0]
                val_tpr = [list(item) for item in val_metrics['tpr']][0]
                tprs_val.append(interp(mean_fpr_val, val_fpr, val_tpr))
                roc_auc = auc(val_fpr, val_tpr)
                aucs_val.append(roc_auc)

                history.add_val_metrics(val_metrics['test_metrics'])

        return history, tprs_train, aucs_train, tprs_val, aucs_val

    def test_gan_bert(self, mode):
        self.transformer.eval()  # maybe redundant
        self.discriminator.eval()
        self.generator.eval()

        total_test_loss = 0
        all_preds = []
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
                model_outputs = self.transformer(b_input_ids, attention_mask=b_input_mask)
                hidden_states = model_outputs[-1]
                _, logits, probs = self.discriminator(hidden_states)
                filtered_logits = logits[:, 0:-1]
                total_test_loss += nll_loss(filtered_logits, b_labels)

            _, preds = torch.max(filtered_logits, 1)

            all_preds += preds.detach().cpu()
            all_labels_ids += b_labels.detach().cpu()

            b_labels = b_labels.cpu().numpy()
            preds = torch.argmax(logits, dim=1).flatten()
            preds = preds.cpu().numpy()

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

    def initialize_model(self, config):
        self.transformer = AutoModel.from_pretrained(self.embedding_model_name)
        model_config = AutoConfig.from_pretrained(self.embedding_model_name)
        hidden_size = int(model_config.hidden_size)

        hidden_levels_g = [hidden_size for i in range(0, config['num_hidden_layers_g'])]
        hidden_levels_d = [hidden_size for i in range(0, config['num_hidden_layers_d'])]

        self.generator = Generator(noise_size=config['noise_size'], output_size=hidden_size,
                                   hidden_sizes=hidden_levels_g, dropout_rate=config['out_dropout_rate'])
        self.discriminator = Discriminator(input_size=hidden_size, hidden_sizes=hidden_levels_d,
                                           num_labels=len(self.label_list), dropout_rate=config['out_dropout_rate'])

        if torch.cuda.is_available():
            self.generator.cuda()
            self.discriminator.cuda()
            self.transformer.cuda()
            if config['multi_gpu']:
                self.transformer = torch.nn.DataParallel(self.transformer)

        transformer_vars = [i for i in self.transformer.parameters()]
        d_vars = transformer_vars + [v for v in self.discriminator.parameters()]
        g_vars = [v for v in self.generator.parameters()]

        self.dis_optimizer = torch.optim.AdamW(d_vars, lr=config['learning_rate_discriminator'])
        self.gen_optimizer = torch.optim.AdamW(g_vars, lr=config['learning_rate_generator'])


class Generator(nn.Module):
    def __init__(self, noise_size=100, output_size=512, hidden_sizes=[512], dropout_rate=0.1):
        super(Generator, self).__init__()
        layers = []
        hidden_sizes = [noise_size] + hidden_sizes
        for i in range(len(hidden_sizes) - 1):
            layers.extend([nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]), nn.LeakyReLU(0.2, inplace=True),
                           nn.Dropout(dropout_rate)])

        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, noise):
        output_rep = self.layers(noise)
        return output_rep


class Discriminator(nn.Module):
    def __init__(self, input_size=512, hidden_sizes=[512], num_labels=2, dropout_rate=0.1):
        super(Discriminator, self).__init__()
        self.input_dropout = nn.Dropout(p=dropout_rate)
        layers = []
        hidden_sizes = [input_size] + hidden_sizes
        for i in range(len(hidden_sizes) - 1):
            layers.extend([nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]), nn.LeakyReLU(0.2, inplace=True),
                           nn.Dropout(dropout_rate)])

        self.layers = nn.Sequential(*layers)  # per il flatten
        self.logit = nn.Linear(hidden_sizes[-1],
                               num_labels + 1)  # +1 for the probability of this sample being fake/real.
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_rep):
        input_rep = self.input_dropout(input_rep)
        last_rep = self.layers(input_rep)
        logits = self.logit(last_rep)
        probs = self.softmax(logits)
        return last_rep, logits, probs
