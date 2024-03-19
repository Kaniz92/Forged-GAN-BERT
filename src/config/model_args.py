import json
import os
from dataclasses import asdict, dataclass, field, fields


@dataclass
class ModelArgs:
    max_seq_length: int = 512
    output_size: int = 2
    hidden_size: int = 300
    embedding_length: int = 100
    mode: str = 'lstm'
    batch_size: int = 8
    apply_balance: bool = True
    epsilon: float = 1e-8
    multi_gpu: bool = True
    apply_scheduler: bool = True
    print_each_n_step: int = 10
    agent_count: int = 1
    model_name: str = "bert-base-cased"
    manual_seed: int = 42
    output_dir: str = 'outputs/'
    optimizer: str = 'adam'
    out_dropout_rate: float = 0.2
    num_train_epochs: int = 10
    warmup_proportion: float = 0.1
    labels_list: list = field(default_factory=list)
    labels_map: dict = field(default_factory=dict)
    lazy_delimiter: str = "\t"
    lazy_labels_column: int = 1
    lazy_loading: bool = False
    lazy_loading_start_line: int = 1
    lazy_text_a_column: bool = None
    lazy_text_b_column: bool = None
    lazy_text_column: int = 0
    onnx: bool = False
    regression: bool = False
    sliding_window: bool = False
    special_tokens_list: list = field(default_factory=list)
    stride: float = 0.8
    tie_value: int = 1

    def load(self, input_dir):
        if input_dir:
            model_args_file = os.path.join(input_dir, "model_args.json")
            if os.path.isfile(model_args_file):
                with open(model_args_file, "r") as f:
                    model_args = json.load(f)

                self.update_from_dict(model_args)


@dataclass
class ModelArgs(ModelArgs):
    model_class: str = 'Model'
