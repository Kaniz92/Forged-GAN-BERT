import argparse

from src.models.deep_sota.deep_model import DeepModel
from src.models.ganbert.ganbert_model import GANBertModel
from src.models.light_gbm.light_gbm_model import LightGBMModel
from src.models.ml_sota.ml_model import MLModel
from src.models.model import Model
from src.models.transformer.transformer_model import TransformerModel
# from src.models.multimodal_transformer.multimodal_transformer_model import MultimodalTransformerModel

MODEL_CLASSES = {
    'default': Model,
    'transformers': TransformerModel,
    'dl': DeepModel,
    'ml': MLModel,
    'gan-bert': GANBertModel,
    'lgb': LightGBMModel,
    # 'multimodal-transformers': MultimodalTransformerModel,
}


def main():
    # TODO: move this to model configs??
    default_model_params = {
        'optimizer': 'adam',
        'out_dropout_rate': 0.2,
        'batch_size': 8,
        'num_train_epochs': 5,
        'warmup_proportion': 0.1,
        'learning_rate': 1e-5,
        'apply_balance': True,
        'num_hidden_layers_g': 1,
        'num_hidden_layers_d': 1,
        'noise_size': 100,
        'apply_scheduler': False,
        'max_seq_length': 64,
        'multi_gpu': True,
        'learning_rate_discriminator': 1e-5,
        'learning_rate_generator': 1e-5,
        'epsilon': 1e-8,
        'print_each_n_step': 10,
        'do_lower_case': True
    }

    parser = argparse.ArgumentParser(description='model params')
    parser.add_argument('--dataset', required=False, help='dataset', default='experiment_1_2_authors_Trial_5')
    parser.add_argument('--dataset_dir', required=False, help='dataset_dir', default='authors_2/trial_1')
    parser.add_argument('--sweep_config_enable', required=False, help='sweep_config_enable', default=False)
    parser.add_argument('--wandb_project_name', required=False, help='wandb_project_name', default="Test_28_11_2022")
    parser.add_argument('--wandb_agent_count', required=False, help='wandb_agent_count', default=50)
    parser.add_argument('--save_model', required=False, help='save_model', default=False)
    parser.add_argument('--output_dir', required=False, help='output_dir', default=None)
    parser.add_argument('--training_strategy', required=False, help='training_strategy', default="wandb")
    parser.add_argument('--model_params', required=False, help='model_params', default=default_model_params)
    parser.add_argument('--evaluate_model', required=False, help='evaluate_model', default=False)
    parser.add_argument('--model_name', required=False, help='model_name', default='cnn')
    parser.add_argument('--model_type', required=False, help='model_type', default='dl')

    parser.add_argument('--embedding_model_name', required=False, help='embedding_model_name',
                        default='bert-base-cased')
    parser.add_argument('--embedding_class', required=False, help='embedding_class',
                        default='Word2Vec')
    # fake_percentage [0, 0.5] since default setting in GAN-BERT is 50% of fake            
    parser.add_argument('--fake_percentage', required=False, help='fake_percentage',
                        default=0.5, type=float)

    args = parser.parse_args()

    dataset = args.dataset
    dataset_dir = args.dataset_dir
    sweep_config_enable = args.sweep_config_enable
    wandb_project_name = args.wandb_project_name
    wandb_agent_count = args.wandb_agent_count
    save_model = args.save_model
    output_dir = args.output_dir
    training_strategy = args.training_strategy
    model_params = args.model_params
    evaluate_model = args.evaluate_model
    model_name = args.model_name
    model_type = args.model_type
    embedding_model_name = args.embedding_model_name
    embedding_class = args.embedding_class
    fake_percentage = args.fake_percentage

    is_Wandb = True

    model = MODEL_CLASSES[model_type]

    model = model(dataset=dataset,
                  dataset_dir=dataset_dir,
                  model_name=model_name,
                  model_type=model_type,
                  sweep_config_enable=sweep_config_enable,
                  wandb_project_name=wandb_project_name,
                  wandb_agent_count=wandb_agent_count,
                  save_model=save_model,
                  output_dir=output_dir,
                  evaluate_model=evaluate_model,
                  is_Wandb=is_Wandb,
                  training_strategy=training_strategy,
                  embedding_model_name=embedding_model_name,
                  embedding_class=embedding_class,
                  fake_percentage = fake_percentage
                  )

    if training_strategy == 'wandb':
        model.train_with_sweep()
    else:
        model.best_model_train(model_params)


if __name__ == "__main__":
    main()
