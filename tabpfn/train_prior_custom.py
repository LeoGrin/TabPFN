import random
import time
import warnings
from datetime import datetime

import torch

import numpy as np

import matplotlib.pyplot as plt
#from scripts.differentiable_pfn_evaluation import eval_model_range
from scripts.model_builder import get_model, get_default_spec, save_model, load_model
from scripts.transformer_prediction_interface import transformer_predict, get_params_from_config, load_model_workflow

from scripts.model_configs import *

from datasets import load_openml_list, open_cc_dids, open_cc_valid_dids
from priors.utils import plot_prior, plot_features
from priors.utils import uniform_int_sampler_f

from scripts.tabular_metrics import calculate_score_per_method, calculate_score
from scripts.tabular_evaluation import evaluate

from priors.differentiable_prior import DifferentiableHyperparameterList, draw_random_style, merge_style_with_info
from scripts import tabular_metrics
from notebook_utils import *
import argparse
import wandb 


parser = argparse.ArgumentParser(description='Train')

parser.add_argument('--prior', type=str, default='mlp')
# n_estimators
parser.add_argument('--n_estimators_lambda', type=float, default=0.15)
parser.add_argument('--n_estimators', type=int, default=None)
# max_depth
parser.add_argument('--max_depth_lambda', type=float, default=0.35)
# checkpoint_file
parser.add_argument('--checkpoint_file', type=str, default=None)

# p_categorical
parser.add_argument('--p_categorical', type=float, default=0.0)

parser.add_argument('--task', type=str, default='multiclass')
parser.add_argument('--wandb', action='store_true')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--name', type=str, default='default')
parser.add_argument('--save_every', type=int, default=20)
# learning rate
parser.add_argument("--lr", type=float, default=0.001)

# correlation proba
parser.add_argument('--correlation_prob', type=float, default=0.0)
parser.add_argument('--correlation_strength', type=float, default=0.0)

args = parser.parse_args()

if args.name == 'default':
    args.name = args.prior + str(random.randint(0, 100000))




large_datasets = True
max_samples = 10000 if large_datasets else 5000
bptt = 10000 if large_datasets else 3000
suite='cc'
device = 'cuda:{}'.format(args.device) if args.device >= 0 else 'cpu'
base_path = '.'
max_features = 100

def print_models(model_string):
    print(model_string)

    for i in range(80):
        for e in range(50):
            exists = Path(os.path.join(base_path, f'models_diff/prior_diff_real_checkpoint{model_string}_n_{i}_epoch_{e}.cpkt')).is_file()
            if exists:
                print(os.path.join(base_path, f'models_diff/prior_diff_real_checkpoint{model_string}_n_{i}_epoch_{e}.cpkt'))
        print()
def train_function(config_sample, i, add_name=''):
    start_time = time.time()
    N_epochs_to_save = 50
    
    def save_callback(model, epoch):
        if not hasattr(model, 'last_saved_epoch'):
            model.last_saved_epoch = 0
        if ((time.time() - start_time) / (maximum_runtime * 60 / N_epochs_to_save)) > model.last_saved_epoch:
            print('Saving model..')
            config_sample['epoch_in_training'] = epoch
            save_model(model, base_path, f'models_diff/prior_diff_real_checkpoint{add_name}_n_{i}_epoch_{model.last_saved_epoch}.cpkt',
                           config_sample)
            model.last_saved_epoch = model.last_saved_epoch + 1 # TODO: Rename to checkpoint
    
    model = get_model(config_sample
                      , device
                      , should_train=True
                      , verbose=1
                      , epoch_callback = save_callback)
    
    return



def reload_config(config_type='causal', task_type='multiclass', longer=0):
    config = get_prior_config(config_type=config_type)
    
    config['prior_type'], config['differentiable'], config['flexible'] = args.prior, True, True
    
    model_string = ''
    
    config['epochs'] = 12000
    config['recompute_attn'] = True

    config['max_num_classes'] = 10
    config['num_classes'] = uniform_int_sampler_f(2, config['max_num_classes'])
    config['balanced'] = True
    model_string = model_string + '_multiclass'
    
    model_string = model_string + '_'+datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    
    return config, model_string

config, model_string = reload_config(longer=1)

config['bptt_extra_samples'] = None

config["sampling"] = "mixed"
del config['differentiable_hyperparameters']['sampling']
config["num_classes"] = 2

if config['prior_type'] == 'trees' or config['prior_type'] == 'mlp_trees':
    config["n_estimators_lambda"] = args.n_estimators_lambda
    config["n_estimators"] = args.n_estimators
    config["max_depth_lambda"] = args.max_depth_lambda
    #config["depth_distribution"] = "uniform"
    #config["split_distribution"] = "uniform"
    #config["split_param"] = 1
elif config['prior_type'] == 'mlp':
    config['pre_sample_causes'] = True

# Not sure what this does
config['output_multiclass_ordered_p'] = 0.
del config['differentiable_hyperparameters']['output_multiclass_ordered_p']

config['multiclass_type'] = 'rank'
del config['differentiable_hyperparameters']['multiclass_type']



config['multiclass_loss_type'] = 'nono' # 'compatible'
config['normalize_to_ranking'] = False # False

config['categorical_feature_p'] = args.p_categorical # diff: .0 #TODO change

# turn this back on in a random search!?
config['nan_prob_no_reason'] = .0
config['nan_prob_unknown_reason'] = .0 # diff: .0
config['set_value_to_nan'] = .1 # diff: 1.

config['normalize_with_sqrt'] = False

config['new_mlp_per_example'] = True
config['prior_mlp_scale_weights_sqrt'] = True
config['batch_size_per_gp_sample'] = None

config['normalize_ignore_label_too'] = False

config['differentiable_hps_as_style'] = False
config['max_eval_pos'] = 1000

config['random_feature_rotation'] = True
config['rotate_normalized_labels'] = True

config["mix_activations"] = False # False heisst eig True

config['emsize'] = 512
config['nhead'] = config['emsize'] // 128
config['bptt'] = 1024+128
config['canonical_y_encoder'] = False

    
config['aggregate_k_gradients'] = 8
config['batch_size'] = 16*config['aggregate_k_gradients']
config['num_steps'] = 1024//config['aggregate_k_gradients']
config['epochs'] = 400
config['total_available_time_in_s'] = None #60*60*22 # 22 hours for some safety...

config['train_mixed_precision'] = True
config['efficient_eval_masking'] = True

config["use_wandb"] = args.wandb
config["name"] = args.name
config["save_every"] = args.save_every

config["lr"] = args.lr

config["correlation_prob"] = args.correlation_strength
config["correlation_strength"] = args.correlation_strength


if args.wandb == True and args.local_rank == 0:
    wandb.init(project="tabpfn_training", entity="leogrin")
    wandb.config.update(config)
    
config_sample = evaluate_hypers(config)

# Load state dict
if args.checkpoint_file:
    path = f"model_checkpoints/{args.checkpoint_file}.pt"
    print(f'Loading checkpoint file {args.checkpoint_file}')
    checkpoint = torch.load(path, map_location=device)

## Training
# launch wandb run
model = get_model(config_sample, device, should_train=True, verbose=1, state_dict=checkpoint if args.checkpoint_file else None)
