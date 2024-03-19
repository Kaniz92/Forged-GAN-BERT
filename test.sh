#!/bin/bash

!PYTHONPATH=./:$PYTHONPATH python run_modal.py \
    --dataset='ChatGPT'\
    --dataset_dir='binary/ACD'\
    --wandb_project_name='chat_gpt_rq1'\
    --training_strategy='best_modal_on_wandb'\
    --model_name='lstm'\
    --model_type='dl'\
    --wandb_agent_count=1
