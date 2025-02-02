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
from priors.utils import uniform_int_sampler_f
import submitit


parser = argparse.ArgumentParser(description='Train')

parser.add_argument('--prior', type=str, default=None)
# n_estimators
parser.add_argument('--n_estimators_lambda', type=float, default=None)
parser.add_argument('--n_estimators', type=int, default=None)
# max_depth
parser.add_argument('--max_depth_lambda', type=float, default=None)
# checkpoint_file
parser.add_argument('--checkpoint_file', type=str, default=None)

parser.add_argument('--num_features', type=int, default=100)

# p_categorical
parser.add_argument('--p_categorical', type=float, default=0.)
# min and max categories
parser.add_argument('--min_categories', type=int, default=2)
parser.add_argument('--max_categories', type=int, default=10)

# correlation proba min and max
parser.add_argument('--correlation_proba_min', type=float, default=0.)
parser.add_argument('--correlation_proba_max', type=float, default=0.)
parser.add_argument('--correlation_strength_min', type=float, default=0.)
parser.add_argument('--correlation_strength_max', type=float, default=0.)
parser.add_argument('--random_feature_removal', type=float, default=0.)
parser.add_argument('--random_feature_removal_min', type=float, default=0.)
parser.add_argument("--n_relevant_features_max", type=int, default=None)
parser.add_argument('--n_relevant_features_min', type=int, default=None)
parser.add_argument('--sampling', type=str, default="mixed")
parser.add_argument('--bptt', type=int, default=1152)
parser.add_argument('--max_eval_pos', type=int, default=1000)
parser.add_argument('--aggregate_k_gradients', type=int, default=8)
parser.add_argument('--get_openml_from_pickle', action='store_true')
parser.add_argument('--curriculum', action='store_true')
parser.add_argument('--curriculum_step', type=int, default=10)
parser.add_argument('--curriculum_tol', type=float, default=0.1)
parser.add_argument('--curriculum_start', type=int, default=5)
parser.add_argument('--criterion_curriculum', type=str, default='relative')
parser.add_argument('--scheduler', type=str, default="cosine")
parser.add_argument("--reset_optim_on_curriculum_step", action='store_true')
parser.add_argument("--num_steps_scheduler", type=int, default=None)
parser.add_argument("--constant_num_features", action='store_true')
parser.add_argument("--eval_prop_num_features", type=float, default=0.5)
parser.add_argument("--sample_bigger_features", type=float, default=0.0)
parser.add_argument("--nhead", type=float, default=4)

parser.add_argument("--test", action='store_true')
parser.add_argument('--num_classes', type=int, default=10)



parser.add_argument('--emsize', default=512, type=int) # sometimes even larger is better e.g. 1024


# whether to return directly the classes instead of the probabilities
parser.add_argument('--return_classes', action='store_true')
#whether to randomize the leaves of the fitted tree before predicting
parser.add_argument('--randomize_leaves', action='store_true')

parser.add_argument('--wandb', action='store_true')
parser.add_argument('--neptune', action='store_true')
parser.add_argument('--offline', action='store_true')
parser.add_argument('--validate_on_datasets', action='store_true')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--name', type=str, default='default')
parser.add_argument('--save_every', type=int, default=30)
# learning rate
parser.add_argument("--lr", type=float, default=0.0001)
# batch size
parser.add_argument("--batch_size", type=int, default=64)
# num steps
parser.add_argument("--num_steps", type=int, default=128)
# num epochs
parser.add_argument("--epochs", type=int, default=400)
# local rank
parser.add_argument("--local_rank", type=int, default=None)
# num_workers
parser.add_argument("--num_workers", type=int, default=10)
parser.add_argument("--partition", type=str, default="gpu")

args = parser.parse_args()

if args.test:
  args.wandb = False
  args.neptune = False
  args.num_steps = 8


if args.name == 'default':
    if args.prior is None:
        args.name = 'default' + str(random.randint(0, 100000))
    else:
        args.name = args.prior + str(random.randint(0, 100000))


device = 'cuda:{}'.format(args.device) if args.device >= 0 else 'cpu'
base_path = '.'

print(f"Using device {device}")

# config = {'lr': 0.0001,
#         'dropout': 0.0,
#         'emsize': 512,
#         'batch_size': 1,
#         'nlayers': 12,
#         'num_features': 100,
#         'nhead': 4,
#         'nhid_factor': 2,
#         'bptt': 10,
#         'eval_positions': [9],
#         'seq_len_used': 50,
#         'sampling': 'mixed',
#         'epochs': 400,
#         'num_steps': 8192,
#         'verbose': False,
#         'mix_activations': True,
#         'nan_prob_unknown_reason_reason_prior': 1.0,
#         'categorical_feature_p': 0.2,
#         'nan_prob_no_reason': 0.0,
#         'nan_prob_unknown_reason': 0.0,
#         'nan_prob_a_reason': 0.0,
#         'max_num_classes': 10,
#         'num_classes': 2,
#         'noise_type': 'Gaussian',
#         'balanced': False,
#         'normalize_to_ranking': False,
#         'set_value_to_nan': 0.1,
#         'normalize_by_used_features': True,
#         'num_features_used': 100, #TODO: check
#         'num_categorical_features_sampler_a': -1.0,
#         'differentiable_hyperparameters': {'prior_bag_exp_weights_1': {'distribution': 'uniform',
#         'min': 1000000.0,
#         'max': 1000001.0},
#         'num_layers': {'distribution': 'meta_trunc_norm_log_scaled',
#         'max_mean': 6,
#         'min_mean': 1,
#         'round': True,
#         'lower_bound': 2},
#         'prior_mlp_hidden_dim': {'distribution': 'meta_trunc_norm_log_scaled',
#         'max_mean': 130,
#         'min_mean': 5,
#         'round': True,
#         'lower_bound': 4},
#         'prior_mlp_dropout_prob': {'distribution': 'meta_beta',
#         'scale': 0.9,
#         'min': 0.1,
#         'max': 5.0},
#         'noise_std': {'distribution': 'meta_trunc_norm_log_scaled',
#         'max_mean': 0.3,
#         'min_mean': 0.0001,
#         'round': False,
#         'lower_bound': 0.0},
#         'init_std': {'distribution': 'meta_trunc_norm_log_scaled',
#         'max_mean': 10.0,
#         'min_mean': 0.01,
#         'round': False,
#         'lower_bound': 0.0},
#         'num_causes': {'distribution': 'meta_trunc_norm_log_scaled',
#         'max_mean': 12,
#         'min_mean': 1,
#         'round': True,
#         'lower_bound': 1},
#         'is_causal': {'distribution': 'meta_choice', 'choice_values': [True, False]},
#         'pre_sample_weights': {'distribution': 'meta_choice',
#         'choice_values': [True, False]},
#         'y_is_effect': {'distribution': 'meta_choice',
#         'choice_values': [True, False]},
#         'prior_mlp_activations': {'distribution': 'meta_choice_mixed',
#         'choice_values': [torch.nn.modules.activation.Tanh,
#             torch.nn.modules.linear.Identity,
#             torch.nn.modules.activation.Tanh,
#             #get_diff_causal,
#             torch.nn.modules.activation.ELU],
#         'choice_values_used': ["<class 'torch.nn.modules.activation.Tanh'>",
#             "<class 'torch.nn.modules.linear.Identity'>",
#             '<function get_diff_causal.<locals>.<lambda> at 0x7fc575dfb670>',
#             "<class 'torch.nn.modules.activation.ELU'>"]},
#         'block_wise_dropout': {'distribution': 'meta_choice',
#         'choice_values': [True, False]},
#         'sort_features': {'distribution': 'meta_choice',
#         'choice_values': [True, False]},
#         'in_clique': {'distribution': 'meta_choice', 'choice_values': [True, False]},
#         'sampling': {'distribution': 'meta_choice',
#         'choice_values': ['normal', 'mixed']},
#         'pre_sample_causes': {'distribution': 'meta_choice',
#         'choice_values': [True, False]},
#         'outputscale': {'distribution': 'meta_trunc_norm_log_scaled',
#         'max_mean': 10.0,
#         'min_mean': 1e-05,
#         'round': False,
#         'lower_bound': 0},
#         'lengthscale': {'distribution': 'meta_trunc_norm_log_scaled',
#         'max_mean': 10.0,
#         'min_mean': 1e-05,
#         'round': False,
#         'lower_bound': 0},
#         'noise': {'distribution': 'meta_choice',
#         'choice_values': [1e-05, 0.0001, 0.01]},
#         'multiclass_type': {'distribution': 'meta_choice',
#         'choice_values': ['value', 'rank']}},
#         'prior_type': 'prior_bag',
#         'differentiable': True,
#         'flexible': True,
#         'aggregate_k_gradients': 8,
#         'recompute_attn': True,
#         'bptt_extra_samples': None,
#         'dynamic_batch_size': False,
#         'multiclass_loss_type': 'nono',
#         'output_multiclass_ordered_p': 0.0,
#         'normalize_with_sqrt': False,
#         'new_mlp_per_example': True,
#         'prior_mlp_scale_weights_sqrt': True,
#         'batch_size_per_gp_sample': None,
#         'normalize_ignore_label_too': True,
#         'differentiable_hps_as_style': False,
#         'max_eval_pos': 1000,
#         'random_feature_rotation': True,
#         'rotate_normalized_labels': True,
#         'canonical_y_encoder': False,
#         'total_available_time_in_s': None,
#         'train_mixed_precision': True,
#         'efficient_eval_masking': True,
#         'multiclass_type': 'rank',
#         'done_part_in_training': 0.8425,
#         'num_features_used_in_training': {'uniform_int_sampler_f(3,max_features)': '<function <lambda>.<locals>.<lambda> at 0x7fc575dfb5e0>'},
#         'num_classes_in_training': '<function <lambda>.<locals>.<lambda> at 0x7fc575dfb550>',
#         'batch_size_in_training': 8,
#         'bptt_in_training': 1024,
#         'bptt_extra_samples_in_training': None,
#         'name': 'default',
#         'use_wandb': False,
#         'save_every': 100}

config = {'lr': 0.0001,
  'dropout': 0.0,
  'emsize': 512,
  'batch_size': 64,
  'nlayers': 12,
  'num_features': 100,
  'nhead': 4,
  'nhid_factor': 2,
  'eval_positions': None,
  'sampling': 'mixed',
  'epochs': 400,
  #'num_steps': 128,
  'verbose': False,
  'pre_sample_causes': True,
  'multiclass_type': 'rank',
  'nan_prob_unknown_reason_reason_prior': 1.0,
  'categorical_feature_p': 0.0, #TODO
  'nan_prob_no_reason': 0.0,
  'nan_prob_unknown_reason': 0.0,
  'nan_prob_a_reason': 0.0,
  'max_num_classes': 10,
  'num_classes': None,
  'noise_type': 'Gaussian',
  'balanced': False,
  'normalize_to_ranking': False,
  'set_value_to_nan': 0.1,
  'normalize_by_used_features': True,
  'num_features_used': {'num_features_func': None},
  'num_categorical_features_sampler_a': -1.0,
  'differentiable_hyperparameters': {'prior_bag_exp_weights_1': {'distribution': 'uniform',
    'min': 1000000.0,
    'max': 1000001.0},
   'num_layers': {'distribution': 'meta_trunc_norm_log_scaled',
    'max_mean': 6,
    'min_mean': 1,
    'round': True,
    'lower_bound': 2},
   'prior_mlp_hidden_dim': {'distribution': 'meta_trunc_norm_log_scaled',
    'max_mean': 130,
    'min_mean': 5,
    'round': True,
    'lower_bound': 4},
   'prior_mlp_dropout_prob': {'distribution': 'meta_beta',
    'scale': 0.9,
    'min': 0.1,
    'max': 5.0},
   'noise_std': {'distribution': 'meta_trunc_norm_log_scaled',
    'max_mean': 0.3,
    'min_mean': 0.0001,
    'round': False,
    'lower_bound': 0.0},
   'init_std': {'distribution': 'meta_trunc_norm_log_scaled',
    'max_mean': 10.0,
    'min_mean': 0.01,
    'round': False,
    'lower_bound': 0.0},
   'num_causes': {'distribution': 'meta_gamma',
    'max_alpha': 3,
    'max_scale': 7,
    'round': True,
    'lower_bound': 2},
   'is_causal': {'distribution': 'meta_choice',
    'choice_values': [True, False]},
   'pre_sample_weights': {'distribution': 'meta_choice',
    'choice_values': [True, False]},
   'y_is_effect': {'distribution': 'meta_choice',
    'choice_values': [True, False]},
   'sampling': {'distribution': 'meta_choice',
    'choice_values': ['normal', 'mixed']},
   'prior_mlp_activations': {'distribution': 'meta_choice_mixed',
    'choice_values': [torch.nn.modules.activation.Tanh,
     torch.nn.modules.linear.Identity]},
   'block_wise_dropout': {'distribution': 'meta_choice',
    'choice_values': [True, False]},
   'sort_features': {'distribution': 'meta_choice',
    'choice_values': [True, False]},
   'in_clique': {'distribution': 'meta_choice',
    'choice_values': [True, False]},
   'outputscale': {'distribution': 'meta_trunc_norm_log_scaled',
    'max_mean': 10.0,
    'min_mean': 1e-05,
    'round': False,
    'lower_bound': 0},
   'lengthscale': {'distribution': 'meta_trunc_norm_log_scaled',
    'max_mean': 10.0,
    'min_mean': 1e-05,
    'round': False,
    'lower_bound': 0},
   'noise': {'distribution': 'meta_choice',
    'choice_values': [1e-05, 0.0001, 0.01]},
   'multiclass_type': {'distribution': 'meta_choice',
    'choice_values': ['value', 'rank']}},
  #'prior_type': 'prior_bag',
  'prior_type': 'mlp',
  'differentiable': True,
  'flexible': True,
  'aggregate_k_gradients': 8,
  'recompute_attn': True,
  'bptt_extra_samples': None,
  'bptt': 1152,
  'dynamic_batch_size': False,
  'multiclass_loss_type': 'nono',
  'output_multiclass_ordered_p': 0.0,
  'normalize_with_sqrt': False,
  'new_mlp_per_example': True,
  'prior_mlp_scale_weights_sqrt': True,
  'batch_size_per_gp_sample': None,
  'normalize_ignore_label_too': False,
  'differentiable_hps_as_style': False,
  'random_feature_rotation': True,
  'rotate_normalized_labels': True,
  'normalize_on_train_only': True,
  'mix_activations': False,
  'weight_decay': 0.0,
  'use_flash_attention': True,
  'canonical_y_encoder': False,
  'total_available_time_in_s': None,
  'train_mixed_precision': True,
  'efficient_eval_masking': True,
  'hardware_batch_size': 4,
  'num_global_att_tokens': 0,
  'use_seperate_decoder': False,
  'attend_to_global_tokens_only_at_test': False,
  "max_eval_pos": 1000}
    


config = {**config, **args.__dict__}
for param_name, param_value in args.__dict__.items():
        print(f"Using {param_name}={param_value}")
  
        
if args.scheduler == "cosine":
  from tabpfn.utils import get_cosine_schedule_with_warmup
  scheduler = get_cosine_schedule_with_warmup
elif args.scheduler == "none":
  from tabpfn.utils import get_no_op_scheduler
  scheduler = get_no_op_scheduler
else:
  raise ValueError(f"Unknown scheduler {args.scheduler}")

if not args.num_steps_scheduler is None:
  scheduler = lambda optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1: get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, args.num_steps_scheduler, num_cycles, last_epoch)
        

config["num_features_no_pad"] = config["num_features"] if not args.curriculum else args.curriculum_start
# a bit confusing but max_num_features is the number of features actually used, while num_feature is the total number including padding
#TODO clean that up because max_num_features is also used elsewhere
config["seq_len_used"] = 50 # I think this has no effect
#config["num_classes"] = 10#uniform_int_sampler_f(2, config['max_num_classes']) #TODO: make it work with return_classes
config["max_num_classes"] = config["num_classes"]
config["num_features_used"] = {'num_features_func': uniform_int_sampler_f(3, config["num_features_no_pad"])} #TODO get rid of differentiable


if args.nhead is None:
  config["nhead"] = config["emsize"] // 64
assert config["nhead"] == int(config["nhead"]), "emsize must be a multiple of 64"
config["nhead"] = int(config["nhead"])
print(f"Using nhead={config['nhead']}")


config["remove_outliers_in_flexible_categorical"] = True
config["normalize_x_in_flexible_categorical"] = True

if args.prior == "linear":
  #already normal
  #TODO: change this when the linear prior becomes more flexible
  config["sampling"] = "normal"
  config["remove_outliers_in_flexible_categorical"] = False
  config["normalize_x_in_flexible_categorical"] = False
  config["random_feature_rotation"] = False



if args.prior is not None:
    assert args.prior in ['trees', 'mlp_trees', "linear"]
    config["prior_type"] = args.prior
    config["n_estimators_lambda"] = args.n_estimators_lambda
    config["n_estimators"] = args.n_estimators
    config["max_depth_lambda"] = args.max_depth_lambda
    config["min_categories"] = args.min_categories
    config["max_categories"] = args.max_categories
    config["assign_class_in_flexible_categorical"] = not args.return_classes
    config["return_classes"] = args.return_classes
    config["randomize_leaves"] = args.randomize_leaves
    print(f"Using {args.prior} prior with n_estimators_lambda={args.n_estimators_lambda}, n_estimators={args.n_estimators}, max_depth_lambda={args.max_depth_lambda}")


    
#config['aggregate_k_gradients'] = 8
#config['batch_size'] = 16*config['aggregate_k_gradients']
#config['num_steps'] = 1024//config['aggregate_k_gradients']
#config['epochs'] = 400
#config['total_available_time_in_s'] = None #60*60*22 # 22 hours for some safety...


config["use_wandb"] = args.wandb
config["use_neptune"] = args.neptune
config["wandb_offline"] = args.offline
config["name"] = args.name
config["save_every"] = args.save_every
config["validate_on_datasets"] =  args.validate_on_datasets


config_sample = evaluate_hypers(config)

# Load state dict
if args.checkpoint_file:
    path = f"model_checkpoints/{args.checkpoint_file}.pt"
    print(f'Loading checkpoint file {args.checkpoint_file}')
    loaded_data = torch.load(path, map_location="cpu")
    print("Length of loaded data", len(loaded_data))
    if len(loaded_data) == 3:
        model_state, optimizer_state, config_sample = loaded_data
    elif len(loaded_data) == 4:
        print('WARNING: Loading model with scheduler state dict')
        model_state = loaded_data["model_state_dict"]
        optimizer_state = loaded_data["optimizer_state_dict"]
        scheduler_state = loaded_data["scheduler_state_dict"]
        epoch = loaded_data["epoch"]
    elif len(loaded_data) == 6:
      model_state = loaded_data["model_state_dict"]
      optimizer_state = loaded_data["optimizer_state_dict"]
      scheduler_state = loaded_data["scheduler_state_dict"]
      epoch = loaded_data["epoch"]
      num_features_no_pad = loaded_data["num_features_no_pad"]
      config_sample_saved = loaded_data["config"]
      # find all mismatch between config_sample and config_sample_saved
      for key, value in config_sample_saved.items():
        if key not in config_sample:
          print(f"WARNING: {key} not in config_sample")
        elif config_sample[key] != value:
          print(f"WARNING: {key} has different value: {config_sample[key]} vs {value}")
      for key, value in config_sample.items():
        if key not in config_sample_saved:
          print(f"WARNING: {key} not in config_sample_saved")
        elif config_sample_saved[key] != value:
          print(f"WARNING: {key} has different value: {config_sample_saved[key]} vs {value}")
    else:
        model_state = loaded_data

## Training
# launch wandb run
#TODO add validation evals
#model = get_model(config_sample, device, should_train=True, verbose=1, state_dict=checkpoint if args.checkpoint_file else None)


executor = submitit.AutoExecutor("slurm")
# partition = "gpu"
# n_gpus = 1
#time max 24h
executor.update_parameters(
    slurm_partition=args.partition,
    gpus_per_node=1,
    timeout_min=60*60,
    
)

job = executor.submit(get_model, config_sample, device, should_train=True, verbose=1, state_dict=model_state if args.checkpoint_file else None,
                      scheduler=scheduler)
