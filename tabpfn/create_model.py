import os
import itertools
import argparse
import time
import datetime
import yaml
from contextlib import nullcontext


import torch
from torch import nn

import tabpfn.utils as utils
from tabpfn.transformer import TransformerModel
from tabpfn.utils import get_cosine_schedule_with_warmup, get_openai_lr, StoreDictKeyPair, get_weighted_single_eval_pos_sampler, get_uniform_single_eval_pos_sampler
import tabpfn.priors as priors
import tabpfn.encoders as encoders
import tabpfn.positional_encodings as positional_encodings
from tabpfn.utils import init_dist
from torch.cuda.amp import autocast, GradScaler
from torch import nn
import wandb
import time
from tqdm import tqdm
import openml
import math
from functools import partial
from tabpfn.priors.utils import trunc_norm_sampler_f, gamma_sampler_f
from tabpfn.utils import get_uniform_single_eval_pos_sampler



class Losses():
    gaussian = nn.GaussianNLLLoss(full=True, reduction='none')
    mse = nn.MSELoss(reduction='none')
    def ce(num_classes):
        num_classes = num_classes.shape[0] if torch.is_tensor(num_classes) else num_classes
        return nn.CrossEntropyLoss(reduction='none', weight=torch.ones(num_classes))
    bce = nn.BCEWithLogitsLoss(reduction='none')
    
def create_dataloader(priordataloader_class, criterion, encoder_generator, emsize=200, nhid=200, nlayers=6, nhead=2, dropout=0.0,
          epochs=10, steps_per_epoch=100, batch_size=200, bptt=10, lr=None, weight_decay=0.0, warmup_epochs=10, input_normalization=False,
          y_encoder_generator=None, pos_encoder_generator=None, decoder=None, extra_prior_kwargs_dict={}, scheduler=get_cosine_schedule_with_warmup,
          load_weights_from_this_state_dict=None, validation_period=10, single_eval_pos_gen=None, bptt_extra_samples=None, gpu_device='cuda:0',
          aggregate_k_gradients=1, verbose=True, style_encoder_generator=None, epoch_callback=None,
          initializer=None, initialize_with_model=None, train_mixed_precision=False, efficient_eval_masking=True, use_wandb=False, name="default", save_every=20,
          num_workers=10, **model_extra_args
          ):
    print(model_extra_args)
    device = gpu_device if torch.cuda.is_available() else 'cpu:0'
    print(f'Using {device} device')
    #print(f'Using {torch.cuda.device_count()} GPUs')
    # batch size
    print(f"Batch size: {batch_size}")
    single_eval_pos_gen = single_eval_pos_gen if callable(single_eval_pos_gen) else lambda: single_eval_pos_gen

    def eval_pos_seq_len_sampler():
        single_eval_pos = single_eval_pos_gen()
        if bptt_extra_samples:
            return single_eval_pos, single_eval_pos + bptt_extra_samples
        else:
            return single_eval_pos, bptt
    get_batch_args = {"num_steps":steps_per_epoch, "batch_size":batch_size, "eval_pos_seq_len_sampler":eval_pos_seq_len_sampler, "seq_len_maximum":bptt+(bptt_extra_samples if bptt_extra_samples else 0), "device":device, **extra_prior_kwargs_dict}
    print("Num workers: ", num_workers)
    dataloader_args = {"num_workers":num_workers, "pin_memory":True, "persistent_workers":True, "prefetch_factor":3}
    #dataloader_args = {"num_workers":0}#10, "pin_memory":True, "persistent_workers":True# "prefetch_factor":3}
    dl = priordataloader_class(dataloader_args, get_batch_args)
    validation_get_batch_args = {"num_steps":64, "batch_size":4, "eval_pos_seq_len_sampler":eval_pos_seq_len_sampler, "seq_len_maximum":bptt+(bptt_extra_samples if bptt_extra_samples else 0), "device":device, **extra_prior_kwargs_dict}
    validation_dataloader_args = {"num_workers":2}
    validation_dl = priordataloader_class(validation_dataloader_args, validation_get_batch_args)
    return dl, validation_dl, device

def create_model(priordataloader_class, criterion, encoder_generator, emsize=200, nhid=200, nlayers=6, nhead=2, dropout=0.0,
          epochs=10, steps_per_epoch=100, batch_size=200, bptt=10, lr=None, weight_decay=0.0, warmup_epochs=10, input_normalization=False,
          y_encoder_generator=None, pos_encoder_generator=None, decoder=None, extra_prior_kwargs_dict={}, scheduler=get_cosine_schedule_with_warmup,
          load_weights_from_this_state_dict=None, validation_period=10, single_eval_pos_gen=None, bptt_extra_samples=None, gpu_device='cuda:0',
          aggregate_k_gradients=1, verbose=True, style_encoder_generator=None, epoch_callback=None,
          initializer=None, initialize_with_model=None, train_mixed_precision=False, efficient_eval_masking=True, use_wandb=False, name="default", save_every=20, 
          num_workers=10, **model_extra_args
          ):
    print("Num workers: ", num_workers)
    dl, validation_dl, device = create_dataloader(priordataloader_class, criterion, encoder_generator, emsize=emsize, nhid=nhid, nlayers=nlayers, nhead=nhead, dropout=dropout,
          epochs=epochs, steps_per_epoch=steps_per_epoch, batch_size=batch_size, bptt=bptt, lr=lr, weight_decay=weight_decay, warmup_epochs=warmup_epochs, input_normalization=input_normalization,
          y_encoder_generator=y_encoder_generator, pos_encoder_generator=pos_encoder_generator, decoder=decoder, extra_prior_kwargs_dict=extra_prior_kwargs_dict, scheduler=scheduler,
          load_weights_from_this_state_dict=load_weights_from_this_state_dict, validation_period=validation_period, single_eval_pos_gen=single_eval_pos_gen, bptt_extra_samples=bptt_extra_samples, gpu_device=gpu_device,
          aggregate_k_gradients=aggregate_k_gradients, verbose=verbose, style_encoder_generator=style_encoder_generator, epoch_callback=epoch_callback,
          initializer=initializer, initialize_with_model=initialize_with_model, train_mixed_precision=train_mixed_precision, efficient_eval_masking=efficient_eval_masking, use_wandb=use_wandb, name=name, save_every=save_every, 
          num_workers=num_workers, **model_extra_args
          )
    

    encoder = encoder_generator(dl.dataset.num_features, emsize)
    #style_def = dl.get_test_batch()[0][0] # the style in batch of the form ((style, x, y), target, single_eval_pos)
    style_def = None
    #print(f'Style definition of first 3 examples: {style_def[:3] if style_def is not None else None}')
    style_encoder = style_encoder_generator(style_def.shape[1], emsize) if (style_def is not None) else None
    if isinstance(criterion, nn.GaussianNLLLoss):
        n_out = 2
    elif isinstance(criterion, nn.CrossEntropyLoss):
        n_out = criterion.weight.shape[0]
    else:
        n_out = 1
    print(f'Number of output units: {n_out}')

    model = TransformerModel(encoder, n_out, emsize, nhead, nhid, nlayers, dropout, style_encoder=style_encoder,
                             y_encoder=y_encoder_generator(1, emsize), input_normalization=input_normalization,
                             pos_encoder=(pos_encoder_generator or positional_encodings.NoPositionalEncoding)(emsize, bptt*2),
                             decoder=decoder, init_method=initializer, efficient_eval_masking=efficient_eval_masking, **model_extra_args
                             )
    model.criterion = criterion

    # load state dict from load_weights_from_this_state_dict
    module_prefix = "module."
    state_dict = {k[len(module_prefix):] if k.startswith(module_prefix) else k: v for k, v in load_weights_from_this_state_dict.items()} if load_weights_from_this_state_dict is not None else None
    
    if state_dict is not None:
        model.load_state_dict(state_dict)
    if initialize_with_model is not None:
        model.init_from_small_model(initialize_with_model)

    print(f"Using a Transformer with {sum(p.numel() for p in model.parameters())/1000/1000:.{2}f} M parameters")

    try:
        for (k, v), (k2, v2) in zip(model.state_dict().items(), initialize_with_model.state_dict().items()):
            print(k, ((v - v2) / v).abs().mean(), v.shape)
    except Exception:
        pass

    model#.to(device)
    
    return model, dl, device, n_out, validation_dl
    
def fix_loaded_config_sample(loaded_config_sample, config):
    def copy_to_sample(*k):
        t,s = loaded_config_sample, config
        for k_ in k[:-1]:
            t = t[k_]
            s = s[k_]
        t[k[-1]] = s[k[-1]]
    copy_to_sample('num_features_used')
    copy_to_sample('num_classes')
    copy_to_sample('differentiable_hyperparameters','prior_mlp_activations','choice_values')

    
def load_config_sample(path, template_config):
    model_state, optimizer_state, loaded_config_sample = torch.load(path, map_location='cpu')
    fix_loaded_config_sample(loaded_config_sample, template_config)
    return loaded_config_sample

def get_default_spec(test_datasets, valid_datasets):
    bptt = 10000
    eval_positions = [1000, 2000, 3000, 4000, 5000] # list(2 ** np.array([4, 5, 6, 7, 8, 9, 10, 11, 12]))
    max_features = max([X.shape[1] for (_, X, _, _, _, _) in test_datasets] + [X.shape[1] for (_, X, _, _, _, _) in valid_datasets])
    max_splits = 5

    return bptt, eval_positions, max_features, max_splits

def get_mlp_prior_hyperparameters(config):
    config = {hp: (list(config[hp].values())[0]) if type(config[hp]) is dict else config[hp] for hp in config}

    if 'random_feature_rotation' not in config:
        config['random_feature_rotation'] = True

    if "prior_sigma_gamma_k" in config:
        sigma_sampler = gamma_sampler_f(config["prior_sigma_gamma_k"], config["prior_sigma_gamma_theta"])
        config['init_std'] = sigma_sampler
    if "prior_noise_std_gamma_k" in config:
        noise_std_sampler = gamma_sampler_f(config["prior_noise_std_gamma_k"], config["prior_noise_std_gamma_theta"])
        config['noise_std'] = noise_std_sampler

    return config


def get_gp_mix_prior_hyperparameters(config):
    return {'lengthscale_concentration': config["prior_lengthscale_concentration"],
            'nu': config["prior_nu"],
            'outputscale_concentration': config["prior_outputscale_concentration"],
            'categorical_data': config["prior_y_minmax_norm"],
            'y_minmax_norm': config["prior_lengthscale_concentration"],
            'noise_concentration': config["prior_noise_concentration"],
            'noise_rate': config["prior_noise_rate"]}

def get_gp_prior_hyperparameters(config):
    return {hp: (list(config[hp].values())[0]) if type(config[hp]) is dict else config[hp] for hp in config}

def get_trees_prior_hyperparameters(config):
    return {hp: (list(config[hp].values())[0]) if type(config[hp]) is dict else config[hp] for hp in config}

def get_mlp_trees_prior_hyperparameters(config):
    return {hp: (list(config[hp].values())[0]) if type(config[hp]) is dict else config[hp] for hp in config}

def get_linear_prior_hyperparameters(config):
    return {hp: (list(config[hp].values())[0]) if type(config[hp]) is dict else config[hp] for hp in config}


def get_meta_gp_prior_hyperparameters(config):
    config = {hp: (list(config[hp].values())[0]) if type(config[hp]) is dict else config[hp] for hp in config}

    if "outputscale_mean" in config:
        outputscale_sampler = trunc_norm_sampler_f(config["outputscale_mean"]
                                                   , config["outputscale_mean"] * config["outputscale_std_f"])
        config['outputscale'] = outputscale_sampler
    if "lengthscale_mean" in config:
        lengthscale_sampler = trunc_norm_sampler_f(config["lengthscale_mean"],
                                                   config["lengthscale_mean"] * config["lengthscale_std_f"])
        config['lengthscale'] = lengthscale_sampler

    return config


def _parse_args(config_parser, parser):
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text

def get_model_no_train(config, device, should_train=True, verbose=False, state_dict=None, epoch_callback=None):
    extra_kwargs = {}
    verbose_train, verbose_prior = verbose >= 1, verbose >= 2
    config['verbose'] = verbose_prior

    if 'aggregate_k_gradients' not in config or config['aggregate_k_gradients'] is None:
        config['aggregate_k_gradients'] = math.ceil(config['batch_size'] * ((config['nlayers'] * config['emsize'] * config['bptt'] * config['bptt']) / 10824640000))

    config['num_steps'] = math.ceil(config['num_steps'] * config['aggregate_k_gradients'])
    config['batch_size'] = math.ceil(config['batch_size'] / config['aggregate_k_gradients'])
    config['recompute_attn'] = config['recompute_attn'] if 'recompute_attn' in config else False

    def make_get_batch(model_proto, **extra_kwargs):
        def new_get_batch(batch_size, seq_len, num_features, hyperparameters
                , device, model_proto=model_proto
                , **kwargs):
            kwargs = {**extra_kwargs, **kwargs} # new args overwrite pre-specified args
            return model_proto.get_batch(
                batch_size=batch_size
                , seq_len=seq_len
                , device=device
                , hyperparameters=hyperparameters
                , num_features=num_features, **kwargs)
        return new_get_batch

    if config['prior_type'] == 'prior_bag':
        # Prior bag combines priors
        get_batch_gp = make_get_batch(priors.fast_gp)
        get_batch_mlp = make_get_batch(priors.mlp)
        if 'flexible' in config and config['flexible']:
            get_batch_gp = make_get_batch(priors.flexible_categorical, **{'get_batch': get_batch_gp})
            get_batch_mlp = make_get_batch(priors.flexible_categorical, **{'get_batch': get_batch_mlp})
        prior_bag_hyperparameters = {'prior_bag_get_batch': (get_batch_gp, get_batch_mlp)
            , 'prior_bag_exp_weights_1': 2.0}
        prior_hyperparameters = {**get_mlp_prior_hyperparameters(config), **get_gp_prior_hyperparameters(config)
            , **prior_bag_hyperparameters}
        model_proto = priors.prior_bag
    else:
        if config['prior_type'] == 'mlp':
            prior_hyperparameters = get_mlp_prior_hyperparameters(config)
            model_proto = priors.mlp
        elif config['prior_type'] == 'gp':
            prior_hyperparameters = get_gp_prior_hyperparameters(config)
            model_proto = priors.fast_gp
        elif config['prior_type'] == 'gp_mix':
            prior_hyperparameters = get_gp_mix_prior_hyperparameters(config)
            model_proto = priors.fast_gp_mix
        elif config['prior_type'] == 'trees':
            prior_hyperparameters = get_trees_prior_hyperparameters(config)
            model_proto = priors.trees
        elif config['prior_type'] == 'mlp_trees':
            prior_hyperparameters = get_mlp_trees_prior_hyperparameters(config)
            model_proto = priors.mlp_trees
        elif config['prior_type'] == 'linear':
            prior_hyperparameters = get_linear_prior_hyperparameters(config)
            model_proto = priors.linear
        else:
            raise Exception()

        if 'flexible' in config and config['flexible']:
            get_batch_base = make_get_batch(model_proto)
            extra_kwargs['get_batch'] = get_batch_base
            model_proto = priors.flexible_categorical

    if config.get('flexible'):
        prior_hyperparameters['normalize_labels'] = True
        prior_hyperparameters['check_is_compatible'] = True
    prior_hyperparameters['prior_mlp_scale_weights_sqrt'] = config[
        'prior_mlp_scale_weights_sqrt'] if 'prior_mlp_scale_weights_sqrt' in prior_hyperparameters else None
    prior_hyperparameters['rotate_normalized_labels'] = config[
        'rotate_normalized_labels'] if 'rotate_normalized_labels' in prior_hyperparameters else True

    use_style = False

    if 'differentiable' in config and config['differentiable']:
        get_batch_base = make_get_batch(model_proto, **extra_kwargs)
        extra_kwargs = {'get_batch': get_batch_base, 'differentiable_hyperparameters': config['differentiable_hyperparameters']}
        model_proto = priors.differentiable_prior
        use_style = True
    print(f"Using style prior: {use_style}")

    if (('nan_prob_no_reason' in config and config['nan_prob_no_reason'] > 0.0) or
        ('nan_prob_a_reason' in config and config['nan_prob_a_reason'] > 0.0) or
        ('nan_prob_unknown_reason' in config and config['nan_prob_unknown_reason'] > 0.0)):
        encoder = encoders.NanHandlingEncoder
    else:
        encoder = partial(encoders.Linear, replace_nan_by_zero=True)

    if config['max_num_classes'] == 2:
        loss = Losses.bce
    elif config['max_num_classes'] > 2:
        loss = Losses.ce(config['max_num_classes'])

    aggregate_k_gradients = 1 if 'aggregate_k_gradients' not in config else config['aggregate_k_gradients']
    check_is_compatible = False if 'multiclass_loss_type' not in config else (config['multiclass_loss_type'] == 'compatible')
    config['multiclass_type'] = config['multiclass_type'] if 'multiclass_type' in config else 'rank'
    config['mix_activations'] = config['mix_activations'] if 'mix_activations' in config else False

    config['bptt_extra_samples'] = config['bptt_extra_samples'] if 'bptt_extra_samples' in config else None
    config['eval_positions'] = [int(config['bptt'] * 0.95)] if config['bptt_extra_samples'] is None else [int(config['bptt'])]

    if "name" not in config:
        config["name"] = "default"
    if "use_wandb" not in config:
        config["use_wandb"] = False
    if "save_every" not in config:
        config["save_every"] = 100

    epochs = 0 if not should_train else config['epochs']
    #print('MODEL BUILDER', model_proto, extra_kwargs['get_batch'])
    print("nhead", config['nhead'])
    model = create_model(model_proto.DataLoader
                , loss
                , encoder
                , name = config['name']
                , use_wandb=config["use_wandb"]
                , save_every=config['save_every']
                , style_encoder_generator = encoders.StyleEncoder if use_style else None
                , emsize=config['emsize']
                , nhead=config['nhead']
                # For unsupervised learning change to NanHandlingEncoder
                , y_encoder_generator= encoders.get_Canonical(config['max_num_classes']) if config.get('canonical_y_encoder', False) else encoders.Linear
                , pos_encoder_generator=None
                , batch_size=config['batch_size']
                , nlayers=config['nlayers']
                , nhid=config['emsize'] * config['nhid_factor']
                , epochs=epochs
                , warmup_epochs=20
                , bptt=config['bptt']
                , gpu_device=device
                , dropout=config['dropout']
                , steps_per_epoch=config['num_steps']
                , single_eval_pos_gen=get_uniform_single_eval_pos_sampler(config.get('max_eval_pos', config['bptt']), min_len=config.get('min_eval_pos', 0))
                , load_weights_from_this_state_dict=state_dict
                , aggregate_k_gradients=aggregate_k_gradients
                , recompute_attn=config['recompute_attn']
                , epoch_callback=epoch_callback
                , bptt_extra_samples = config['bptt_extra_samples']
                , extra_prior_kwargs_dict={
            'num_features': config['num_features']
            , 'hyperparameters': prior_hyperparameters
            #, 'dynamic_batch_size': 1 if ('num_global_att_tokens' in config and config['num_global_att_tokens']) else 2
            , 'batch_size_per_gp_sample': config.get('batch_size_per_gp_sample', None)
            , **extra_kwargs
        }
                , lr=config['lr']
                , verbose=verbose_train,
                weight_decay=config.get('weight_decay', 0.0))[0]

    #return total_loss, total_positional_losses, model.to('cpu'), dl
    # imitate train function return values
    return 0, 0, model.to('cpu'), None

def load_model_no_train(path, filename, device, config_sample=None, verbose=0):
    # TODO: This function only restores evaluation functionality but training canät be continued. It is also not flexible.
    # print('Loading....')
    loaded_data = torch.load(
        os.path.join(path, filename), map_location='cpu')
    print("Length of loaded data: ", len(loaded_data))
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
        # transfer config values
        for k in config_sample_saved:
            if type(config_sample_saved[k]) == dict and len(config_sample_saved[k]) == 0:
                print('WARNING: Config key {} has no value'.format(k))
            else:
                config_sample[k] = config_sample_saved[k]
    else:
        model_state = loaded_data


    
        
    if ('differentiable_hyperparameters' in config_sample
            and 'prior_mlp_activations' in config_sample['differentiable_hyperparameters']):
        config_sample['differentiable_hyperparameters']['prior_mlp_activations']['choice_values_used'] = config_sample[
                                                                                                         'differentiable_hyperparameters'][
                                                                                                         'prior_mlp_activations'][
                                                                                                         'choice_values']
        config_sample['differentiable_hyperparameters']['prior_mlp_activations']['choice_values'] = [
            torch.nn.Tanh for k in config_sample['differentiable_hyperparameters']['prior_mlp_activations']['choice_values']]

    # deduce the model parameters from the model state dict
    module_prefix = 'module.'
    model_state = {k.replace(module_prefix, ''): v for k, v in model_state.items()}
    config_sample["emsize"] = model_state['encoder.weight'].shape[0]
    config_sample["nlayers"] = len([k for k in model_state.keys() if k.startswith('transformer_encoder.layers') and k.endswith('self_attn.out_proj.weight')])
    config_sample["nhid_factor"] = model_state['transformer_encoder.layers.0.linear1.weight'].shape[0] // config_sample["emsize"]
    hidden_size = config_sample["emsize"] * config_sample["nhid_factor"]
    config_sample["nhead"] = config_sample["nhead"] # not sure if it can be deduced from the state dict
    print(f"WARNING: Using nhead={config_sample['nhead']} from config, not from the model state dict. This can fail silently.")
    config_sample["num_features"] =  model_state['encoder.weight'].shape[1]
    print(f"Loaded model with parameters: emsize={config_sample['emsize']}, nhead={config_sample['nhead']}, nlayers={config_sample['nlayers']}, nhid_factor={config_sample['nhid_factor']}, num_features={config_sample['num_features']}")
    config_sample['categorical_features_sampler'] = lambda: lambda x: ([], [], [])
    config_sample['num_features_used_in_training'] = config_sample['num_features_used']
    config_sample['num_features_used'] = lambda: config_sample['num_features']
    config_sample['num_classes_in_training'] = config_sample['num_classes']
    config_sample['num_classes'] = config_sample['num_classes']
    config_sample['batch_size_in_training'] = config_sample['batch_size']
    config_sample['batch_size'] = 1
    config_sample['bptt_in_training'] = config_sample['bptt']
    config_sample['bptt'] = 10
    config_sample['bptt_extra_samples_in_training'] = config_sample['bptt_extra_samples']
    config_sample['bptt_extra_samples'] = None

    #print('Memory', str(get_gpu_memory()))

    model = get_model_no_train(config_sample, device=device, should_train=False, verbose=verbose)
    model[2].load_state_dict(model_state)
    model[2].to(device)
    model[2].eval()

    return model, config_sample

def load_model_no_train_from_pytorch(model, config_sample):
    # TODO: This function only restores evaluation functionality but training canät be continued. It is also not flexible.
    # print('Loading....')
    #print('Memory', str(get_gpu_memory()))
    import copy
    model_ = copy.deepcopy(model)
    
    if ('differentiable_hyperparameters' in config_sample and 'prior_mlp_activations' in config_sample['differentiable_hyperparameters']):
        config_sample['differentiable_hyperparameters']['prior_mlp_activations']['choice_values_used'] = config_sample[
                                                                                                        'differentiable_hyperparameters'][
                                                                                                        'prior_mlp_activations'][
                                                                                                        'choice_values']
        config_sample['differentiable_hyperparameters']['prior_mlp_activations']['choice_values'] = [
            torch.nn.Tanh for k in config_sample['differentiable_hyperparameters']['prior_mlp_activations']['choice_values']]

    config_sample['categorical_features_sampler'] = lambda: lambda x: ([], [], [])
    config_sample['num_features_used_in_training'] = config_sample['num_features_used']
    config_sample['num_features_used'] = lambda: config_sample['num_features']
    config_sample['num_classes_in_training'] = config_sample['num_classes']
    config_sample['num_classes'] = 2
    config_sample['batch_size_in_training'] = config_sample['batch_size']
    config_sample['batch_size'] = 1
    config_sample['bptt_in_training'] = config_sample['bptt']
    config_sample['bptt'] = 10 #WHY
    config_sample['bptt_extra_samples_in_training'] = config_sample['bptt_extra_samples']
    config_sample['bptt_extra_samples'] = None

    model = 0, 0, model_, None
    #module_prefix = 'module.'
    module_prefix = ""
    model_state = {k.replace(module_prefix, ''): v for k, v in model[2].state_dict().items()}
    model[2].load_state_dict(model_state)
    #model[2].to(device)
    model[2].eval()

    return model, config_sample
