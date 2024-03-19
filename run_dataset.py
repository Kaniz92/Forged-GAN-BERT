import argparse

from src.data.data_split import DataSplitModule
from src.data.generate_data import DataGeneratorModule

def main():
    parser = argparse.ArgumentParser(description='model params')
    parser.add_argument('--dataset', required=False, help='dataset', default='experiment_1_2_authors_Trial_5')
    parser.add_argument('--dataset_dir', required=False, help='dataset_dir', default='authors_2/trial_1')
    parser.add_argument('--model_id', required=False, help='model_id', default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument('--dataset_action', required=False, help='dataset_action', default='split')
    parser.add_argument('--model_name', required=False, help='model_name', default='llama')
    parser.add_argument('--prompttype', required=False, help='prompttype', default='Default')
    parser.add_argument('--category', required=False, help='category', default='None')
    parser.add_argument('--temperature', required=False, help='temperature', default=1)
    
    args = parser.parse_args()

    dataset = args.dataset
    dataset_dir = args.dataset_dir
    model_id = args.model_id
    dataset_action = args.dataset_action
    model_name = args.model_name
    prompttype = args.prompttype
    category = args.category
    temperature = int(args.temperature)

    if dataset_action == 'split':
        dataSplitModule = DataSplitModule(dataset, dataset_dir)
        dataSplitModule.split_and_save_data()
    elif dataset_action == 'generate':
        dataGeneratorModule = DataGeneratorModule(model_id, model_name, dataset, dataset_dir, prompttype, category, temperature)
        dataGeneratorModule.generate_novels()
        
if __name__ == "__main__":
    main()
