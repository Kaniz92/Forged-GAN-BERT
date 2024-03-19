import numpy as np
from sklearn.metrics import f1_score, precision_score, roc_curve, recall_score, accuracy_score, classification_report


class MetricsHistory:
    def __init__(self):
        self.train_metrics = Metrics()
        self.val_metrics = Metrics()
        self.test_metrics = Metrics()
        self.test_1_metrics = Metrics()
        self.test_0_metrics = Metrics()

    def add_train_metrics(self, values):
        self.train_metrics.append_values(values)

    def add_val_metrics(self, values):
        self.val_metrics.append_values(values)

    def add_test_metrics(self, values):
        self.test_metrics.append_values(values)

    def add_test_1_metrics(self, values):
        self.test_1_metrics.append_class_based_values(values)

    def add_test_0_metrics(self, values):
        self.test_0_metrics.append_class_based_values(values)

    @staticmethod
    def calculate_average(metric_type):
        return metric_type.calculate_average()


class Metrics:
    def __init__(self):
        self.accuracy = []
        self.f1 = []
        self.precision = []
        self.recall = []
        self.support = []

    def append_values(self, values):
        self.accuracy.append(values['accuracy'])
        self.f1.append(values['f1'])
        self.precision.append(values['precision'])
        self.recall.append(values['recall'])

    def append_class_based_values(self, values):
        self.support.append(values['support'])
        self.f1.append(values['f1'])
        self.precision.append(values['precision'])
        self.recall.append(values['recall'])

    def calculate_average(self):
        return np.mean(self.accuracy), np.mean(self.f1), np.mean(self.precision), np.mean(self.recall), np.mean(
            self.support)


def calculate_metrics(labels, predictions, pos_label):
    fpr, tpr, thresholds = roc_curve(labels, predictions, pos_label=pos_label)

    metrics = {'accuracy': accuracy_score(labels, predictions), 'f1': f1_score(labels, predictions, average='macro'),
               'precision': precision_score(labels, predictions, average='macro'),
               'recall': recall_score(labels, predictions, average='macro'),
               'fpr': fpr,
               'tpr': tpr,
               'thresholds': thresholds}

    return metrics


def calculate_classification_report(labels, predictions, pos_label, class_index):
    report_dict = classification_report(labels, predictions, output_dict=True)
    fpr, tpr, thresholds = roc_curve(labels, predictions, pos_label=pos_label)

    metrics = {
        'test_metrics': {
            'accuracy': report_dict['accuracy'],
            'f1': report_dict['macro avg']['f1-score'],
            'precision': report_dict['macro avg']['precision'],
            'recall': report_dict['macro avg']['recall']
        },
        'test_0_metrics': {
            'f1': report_dict['0']['f1-score'] if '0' in report_dict else 0.000,
            'precision': report_dict['0']['precision'] if '0' in report_dict else 0.000,
            'recall': report_dict['0']['recall'] if '0' in report_dict else 0.000,
            'support': report_dict['0']['support'] if '0' in report_dict else 0.000,
        },
        'test_1_metrics': {
            'f1': report_dict[class_index]['f1-score'] if class_index in report_dict else 0.000,
            'precision': report_dict[class_index]['precision'] if class_index in report_dict else 0.000,
            'recall': report_dict[class_index]['recall'] if class_index in report_dict else 0.000,
            'support': report_dict[class_index]['support'] if class_index in report_dict else 0.000,
        },
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds}

    return metrics
