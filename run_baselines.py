import argparse
import random
import numpy as np
import torch
import os

from src.models.baselines import BaselineModule

def seed_everything(seed=42):
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser(description='model params')
    parser.add_argument('--dataset_dir', required=False, help='dataset_dir', default='authors_2/trial_1')
    parser.add_argument('--is_local', required=False, help='is_local', default=False)

    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    is_local = args.is_local

    seed_everything()

    dataset_names = ['ACD', 'HRD', 'JL', 'MT', 'WK']
    for dataset_name in dataset_names:
        print(f'############ Processing author: {dataset_name} ############')
        baselineModule = BaselineModule(dataset_name, dataset_dir, is_local)
        baselineModule.calculate_metrics()
#         get results to csv later

if __name__ == "__main__":
    main()
