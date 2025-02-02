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
from tabpfn.scripts.transformer_prediction_interface import TabPFNClassifier
from evaluate_model import get_validation_performance
from create_model import create_model, load_model_no_train_from_pytorch, create_dataloader
import neptune
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
import numpy as np
from priors.utils import uniform_int_sampler_f

def remove_callables(config):
    clean_config = {}
    for key, value in config.items():
        if not callable(value):
            if isinstance(value, dict):
                clean_config[key] = remove_callables(value)
            else:
                clean_config[key] = value
    return clean_config


class Losses():
    gaussian = nn.GaussianNLLLoss(full=True, reduction='none')
    mse = nn.MSELoss(reduction='none')
    def ce(num_classes):
        num_classes = num_classes.shape[0] if torch.is_tensor(num_classes) else num_classes
        return nn.CrossEntropyLoss(reduction='none', weight=torch.ones(num_classes))
    bce = nn.BCEWithLogitsLoss(reduction='none')
    



def train(priordataloader_class, criterion, encoder_generator, emsize=200, nhid=200, nlayers=6, nhead=2, dropout=0.0,
          epochs=10, steps_per_epoch=100, batch_size=200, bptt=10, lr=None, weight_decay=0.0, warmup_epochs=10, input_normalization=False,
          y_encoder_generator=None, pos_encoder_generator=None, decoder=None, extra_prior_kwargs_dict={}, scheduler_func=get_cosine_schedule_with_warmup,
          load_weights_from_this_state_dict=None, validation_period=10, single_eval_pos_gen=None, bptt_extra_samples=None, gpu_device='cuda:0',
          aggregate_k_gradients=1, verbose=True, style_encoder_generator=None, epoch_callback=None,
          initializer=None, initialize_with_model=None, train_mixed_precision=False, efficient_eval_masking=True, use_wandb=False, 
          wandb_offline=False, use_neptune=False, validate_on_datasets=False, name="default", save_every=20, config={}, 
          get_openml_from_pickle=False, curriculum=False, curriculum_tol=0.1, curriculum_step=1, curriculum_start=5, **model_extra_args
          ):
    model, dl, device, n_out, validation_dl = create_model(priordataloader_class, criterion, encoder_generator, emsize, nhid, nlayers, nhead, dropout,
                                                       epochs, steps_per_epoch, batch_size, bptt, lr, weight_decay, warmup_epochs, input_normalization,
                                                       y_encoder_generator, pos_encoder_generator, decoder, extra_prior_kwargs_dict, scheduler_func,
                                                       load_weights_from_this_state_dict, validation_period, single_eval_pos_gen, bptt_extra_samples, gpu_device,
                                                       aggregate_k_gradients, verbose, style_encoder_generator, epoch_callback,
                                                       initializer, initialize_with_model, train_mixed_precision, efficient_eval_masking, use_wandb, name, save_every, 
                                                       config["num_workers"], **model_extra_args)
    # using_dist, rank, device = init_dist(device)
    # print("Using distributed training:", using_dist)
    # if using_dist:
    #     print("Distributed training")
    #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank, broadcast_buffers=False)
    # dl.model = model
    using_dist, rank, device = init_dist(device)
    #torch.cuda.set_device(rank)
    #gpu = torch.device("cuda")
    model = model.to(device)
    print("Using distributed training:", using_dist)
    if using_dist:
        print("Distributed training")
        #model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank, broadcast_buffers=False)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    dl.model = model
    


    # learning rate
    if lr is None:
        lr = get_openai_lr(model)
        print(f"Using OpenAI max lr of {lr}.")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    print("scheduler", scheduler_func)
    scheduler = scheduler_func(optimizer, warmup_epochs, epochs if epochs is not None else 100) # when training for fixed time lr schedule takes 100 steps

    scaler = GradScaler() if train_mixed_precision else None

    # check that everything uses up-to-date APIs
    utils.check_compatibility(dl)
    
    if use_wandb and rank == 0:
        if wandb_offline:
            print("setting wandb offline")
            os.environ['WANDB_MODE'] = 'offline'
        print("initializing wandb")
        wandb.init(project="tabpfn_training-3", entity="leogrin")
        wandb.config.update(config)
        print("wandb initialized")
        print("name", name)
        print("wandb id", wandb.run.id)
        name += "_" + wandb.run.id
    if use_neptune and rank == 0:
        print("initializing neptune")
        if wandb_offline:
            print("setting neptune offline")
            run = neptune.init_run(
                project="leogrin/tabpfn-training",
                api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkNGY1NTJlYS0wNzgzLTQxM2EtOWFlZi02NmVmNmQ2MTRlNWIifQ==",
                mode="offline"
            )
        else:
            print("setting neptune online")
            run = neptune.init_run(
                project="leogrin/tabpfn-training",
                api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkNGY1NTJlYS0wNzgzLTQxM2EtOWFlZi02NmVmNmQ2MTRlNWIifQ=="
            )
        config["wandb_id"] = wandb.run.id
        run["config"] = config

    
    def train_epoch(dl, validation_dl):
        
        model.train()  # Turn on the train mode
        total_loss = 0.
        total_positional_losses = 0.
        total_positional_losses_recorded = 0
        nan_steps = 0
        ignore_steps = 0
        before_get_batch = time.time()
        assert len(dl) % aggregate_k_gradients == 0, 'Please set the number of steps per epoch s.t. `aggregate_k_gradients` divides it.'
        pbar = tqdm(enumerate(dl))
        print("len(dl)", len(dl))
        for batch, (data, targets, single_eval_pos) in pbar:
            print("data", data)
            print("targets", targets)
            if using_dist and not (batch % aggregate_k_gradients == aggregate_k_gradients - 1):
                cm = model.no_sync()
            else:
                cm = nullcontext()
            with cm:
                time_to_get_batch = time.time() - before_get_batch
                before_forward = time.time()
                if bptt_extra_samples is None:
                    single_eval_pos = single_eval_pos_gen() if callable(single_eval_pos_gen) else single_eval_pos_gen
                else:
                    single_eval_pos = targets.shape[0] - bptt_extra_samples

                with autocast(enabled=scaler is not None):
                    # If style is set to None, it should not be transferred to device
                    
                    output = model(tuple(e.to(device, non_blocking=True) if torch.is_tensor(e) else e for e in data) if (isinstance(data, tuple) or isinstance(data, list)) else data.to(device, non_blocking=True)
                                   , single_eval_pos=single_eval_pos)

                    forward_time = time.time() - before_forward

                    if single_eval_pos is not None:
                        targets = targets[single_eval_pos:]
                    if isinstance(criterion, nn.GaussianNLLLoss):
                        assert output.shape[-1] == 2, \
                            'need to write a little bit of code to handle multiple regression targets at once'

                        mean_pred = output[..., 0]
                        var_pred = output[..., 1].abs()
                        losses = criterion(mean_pred.flatten(), targets.to(device, non_blocking=True).flatten(), var=var_pred.flatten())
                    elif isinstance(criterion, (nn.MSELoss, nn.BCEWithLogitsLoss)):
                        print("here")
                        losses = criterion(output.flatten(), targets.to(device, non_blocking=True).flatten())
                    elif isinstance(criterion, nn.CrossEntropyLoss):
                        losses = criterion(output.reshape(-1, n_out), targets.to(device, non_blocking=True).long().flatten())
                    else:
                        losses = criterion(output, targets)
                    losses = losses.view(*output.shape[0:2])
                    loss, nan_share = utils.torch_nanmean(losses.mean(0), return_nanshare=True)
                    loss = loss / aggregate_k_gradients
                    

                    

                if scaler: loss = scaler.scale(loss)
                before_backward = time.time()
                loss.backward()

                if batch % aggregate_k_gradients == aggregate_k_gradients - 1:
                    if scaler: scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    try:
                        if scaler:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                    except:
                        print("Invalid optimization step encountered")
                    optimizer.zero_grad()
                backward_time = time.time() - before_backward

                step_time = time.time() - before_forward

                pbar.set_description(f"Loss: {loss.item():.4f}, nan_share: {nan_share:.4f}, ignore_steps: {ignore_steps}, nan_steps: {nan_steps}")

                if not torch.isnan(loss):
                    total_loss += losses.mean().cpu().detach().item()
                    if batch % 20 == 0:
                        print("Batch loss: ", losses.mean().cpu().detach().item())
                        print(f"Batch {batch} took {step_time:.{2}f}s to process, {time_to_get_batch:.{2}f}s to get batch, {forward_time:.{2}f}s to forward")
                    total_positional_losses += losses.mean(1).cpu().detach() if single_eval_pos is None else \
                        nn.functional.one_hot(torch.tensor(single_eval_pos), bptt)*\
                        losses[:bptt-single_eval_pos].mean().cpu().detach()
                    total_positional_losses_recorded += torch.ones(bptt) if single_eval_pos is None else \
                        nn.functional.one_hot(torch.tensor(single_eval_pos), bptt)
                nan_steps += nan_share

                ignore_steps += (targets == -100).float().mean()

            before_get_batch = time.time()
        print("Epoch Loss", total_loss / steps_per_epoch)
        # if prior is linear, compute the loss for the lasso
        print("Computing lasso vs tabpfn")
        try:
            pbar_val = tqdm(enumerate(validation_dl))
            print("len(validation_dl)", len(validation_dl))
            balanced_acc_tabpfn_adjusted_list = []
            balanced_acc_lasso_adjusted_list = []
            balanced_acc_tabpfn_list = []
            balanced_acc_lasso_list = []
            val_loss_list = []
            for batch, (data, targets, single_eval_pos) in pbar_val:
                with torch.no_grad():
                    
                    if single_eval_pos is not None:
                        targets = targets[single_eval_pos:]
                        
                    output = model(tuple(e.to(device, non_blocking=True) if torch.is_tensor(e) else e for e in data) if (isinstance(data, tuple) or isinstance(data, list)) else data.to(device, non_blocking=True)
                                    , single_eval_pos=single_eval_pos)
                    if isinstance(criterion, nn.GaussianNLLLoss):
                        assert output.shape[-1] == 2, \
                            'need to write a little bit of code to handle multiple regression targets at once'
                        mean_pred = output[..., 0]
                        var_pred = output[..., 1].abs()
                        val_losses = criterion(mean_pred.flatten(), targets.to(device, non_blocking=True).flatten(), var=var_pred.flatten())
                    elif isinstance(criterion, (nn.MSELoss, nn.BCEWithLogitsLoss)):
                        print("here")
                        val_losses = criterion(output.flatten(), targets.to(device, non_blocking=True).flatten())
                    elif isinstance(criterion, nn.CrossEntropyLoss):
                        val_losses = criterion(output.reshape(-1, n_out), targets.to(device, non_blocking=True).long().flatten())
                    else:
                        val_losses = criterion(output, targets)
                    val_losses = val_losses.view(*output.shape[0:2])
                    val_loss, nan_share = utils.torch_nanmean(val_losses.mean(0), return_nanshare=True)
                    val_loss_list.append(val_loss.cpu().detach().item())
                    lasso = LogisticRegression(penalty='l1', max_iter=1000, solver='liblinear')
                    X_cpu = data[1].cpu().numpy()
                    y_cpu = data[2].cpu().numpy()

                    for i in range(X_cpu.shape[1]):
                        X = X_cpu[:, i, :]
                        y = y_cpu[:, i]
                        print("X.shape", X.shape)
                        print(np.unique(y, return_counts=True))           
                        n_zero_cols = len(np.where(~X.any(axis=0))[0])
                        print("num zero_cols", n_zero_cols)
                        actual_num_features_no_pad = X.shape[1] - n_zero_cols
                        if actual_num_features_no_pad < config["eval_prop_num_features"] * extra_prior_kwargs_dict["hyperparameters"]['num_features_no_pad']:
                            print(f"Number of features in the prior is too small, skipping. Actual number of features: {actual_num_features_no_pad}, required number of features: {config['eval_prop_num_features'] * extra_prior_kwargs_dict['hyperparameters']['num_features_no_pad']}")
                            continue
                        X_train, X_test = X[:single_eval_pos], X[single_eval_pos:]
                        y_train, y_test = y[:single_eval_pos], y[single_eval_pos:]
                        if len(np.unique(y_train)) == 1:
                            print("All samples in the training set belong to the same class, skipping")
                            continue
                        lasso.fit(X_train, y_train)
                        # compute balanced accuracy
                        preds = lasso.predict(X_test)
                        balanced_acc_adjusted = balanced_accuracy_score(y_test, preds, adjusted=True)
                        balanced_acc = balanced_accuracy_score(y_test, preds)
                        preds_tabpfn = output[:, i, :].detach().cpu().numpy().argmax(axis=1)
                        # try to predict with batch 1
                        balanced_acc_tabpfn_adjusted = balanced_accuracy_score(y_test, preds_tabpfn, adjusted=True)
                        balanced_acc_tabpfn = balanced_accuracy_score(y_test, preds_tabpfn)
                        balanced_acc_tabpfn_adjusted_list.append(balanced_acc_tabpfn_adjusted)
                        balanced_acc_lasso_adjusted_list.append(balanced_acc_adjusted)
                        balanced_acc_tabpfn_list.append(balanced_acc_tabpfn)
                        balanced_acc_lasso_list.append(balanced_acc)
            mean_diff_balanced_acc_adjusted = np.mean(np.array(balanced_acc_lasso_adjusted_list) - np.array(balanced_acc_tabpfn_adjusted_list))
            balanced_acc_lasso_adjusted = np.mean(balanced_acc_lasso_adjusted_list)
            balanced_acc_tabpfn_adjusted = np.mean(balanced_acc_tabpfn_adjusted_list)
            mean_relative_diff_balanced_acc = np.mean((np.array(balanced_acc_lasso_list) - np.array(balanced_acc_tabpfn_list)) / np.array(balanced_acc_lasso_list))
            mean_val_loss = np.mean(val_loss_list)
            
            print("Balanced accuracy tabpfn (adjusted): ", balanced_acc_tabpfn_adjusted)
            print("Balanced accuracy lasso (adjusted): ", balanced_acc_lasso_adjusted)
            print("Mean difference balanced accuracy adjusted: ", mean_diff_balanced_acc_adjusted)
            print("Balanced accuracy tabpfn: ", np.mean(balanced_acc_tabpfn_list))
            print("Balanced accuracy lasso: ", np.mean(balanced_acc_lasso_list))
            print("Mean relative difference balanced accuracy: ", mean_relative_diff_balanced_acc)
            print("Mean validation loss: ", mean_val_loss)
        except:
            print("Could not compute lasso")
            mean_diff_balanced_acc_adjusted = np.nan
            balanced_acc_tabpfn_adjusted = np.nan
            mean_relative_diff_balanced_acc = np.nan
            balanced_acc_lasso_adjusted = np.nan
            mean_val_loss = np.nan
            
        
        return total_loss / steps_per_epoch, (total_positional_losses / total_positional_losses_recorded).tolist(),\
            time_to_get_batch, forward_time, step_time, nan_steps.cpu().item()/(batch+1),\
            ignore_steps.cpu().item()/(batch+1), balanced_acc_tabpfn_adjusted, mean_diff_balanced_acc_adjusted, mean_relative_diff_balanced_acc, balanced_acc_lasso_adjusted, mean_val_loss

    total_loss = float('inf')
    total_positional_losses = float('inf')
    try:
        for epoch in (range(1, epochs + 1) if epochs is not None else itertools.count(1)):

            epoch_start_time = time.time()
            total_loss, total_positional_losses, time_to_get_batch, forward_time, step_time, nan_share,\
            ignore_share, balanced_acc_tabpfn, mean_diff_balanced_acc, mean_relative_diff_balanced_acc, balanced_acc_lasso, mean_val_loss =\
                train_epoch(dl, validation_dl)
            if curriculum:
                #TODO make this more flexible
                if config["criterion_curriculum"] == "relative":
                    condition = mean_relative_diff_balanced_acc < curriculum_tol
                elif config["criterion_curriculum"] == "absolute":
                    condition = mean_diff_balanced_acc < curriculum_tol
                if condition:
                    print("Mean relative difference balanced accuracy is below tolerance, increasing number of features")
                    print("Mean relative difference balanced accuracy: ", mean_relative_diff_balanced_acc)
                    print("Max number of features: ", extra_prior_kwargs_dict["hyperparameters"]['num_features'])
                    new_max_num_features = min(extra_prior_kwargs_dict["hyperparameters"]['num_features_no_pad'] + curriculum_step, extra_prior_kwargs_dict['num_features'])
                    print("Going from ", extra_prior_kwargs_dict["hyperparameters"]['num_features_no_pad'], " to ", new_max_num_features)
                    extra_prior_kwargs_dict["hyperparameters"]['num_features_no_pad'] = new_max_num_features
                    #extra_prior_kwargs_dict['num_features_used'] = {'num_features_func': uniform_int_sampler_f(3, new_max_num_features)}
                    dl, validation_dl, _ = create_dataloader(priordataloader_class, criterion, encoder_generator, emsize, nhid, nlayers, nhead, dropout,
                                                    epochs, steps_per_epoch, batch_size, bptt, lr, weight_decay, warmup_epochs, input_normalization,
                                                    y_encoder_generator, pos_encoder_generator, decoder, extra_prior_kwargs_dict, scheduler,
                                                    load_weights_from_this_state_dict, validation_period, single_eval_pos_gen, bptt_extra_samples, gpu_device,
                                                    aggregate_k_gradients, verbose, style_encoder_generator, epoch_callback,
                                                    initializer, initialize_with_model, train_mixed_precision, efficient_eval_masking, use_wandb, name, save_every, **model_extra_args)
                    if config["reset_optim_on_curriculum_step"]:
                        print("Resetting optimizer and scheduler")
                        if lr is None:
                            lr = get_openai_lr(model)
                            print(f"Using OpenAI max lr of {lr}.")
                        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
                        print("scheduler", scheduler)
                        scheduler = scheduler_func(optimizer, warmup_epochs, epochs if epochs is not None else 100) # when training for fixed time lr schedule takes 100 steps

                    
            if hasattr(dl, 'validate') and epoch % validation_period == 0:
                with torch.no_grad():
                    val_score = dl.validate(model)
            else:
                val_score = None

            if (use_wandb or use_neptune) and rank == 0:
                model_sklearn = TabPFNClassifier(no_preprocess_mode=True, device=device)
                model_pytorch = load_model_no_train_from_pytorch(model, config_sample=model_sklearn.c)[0]
                model_sklearn.model = model_pytorch
                if validate_on_datasets:
                    try:
                        import idr_torch
                        on_jean_zay = True
                        print("We're on Jean Zay, loading the datasets from pickle")
                    except:
                        on_jean_zay = False
                        print("We're not on Jean Zay, loading the datasets from openml")
                    if on_jean_zay or get_openml_from_pickle:
                        measure_on_datasets = get_validation_performance(model_sklearn, datasets = [
                                [44089, 44120, 44121, 44122, 44123, 44125, 44126, 44128, 44129, 44130, 45022,
                                    45021, 45020, 45019, 45028, 45026],
                                [44156, 44157, 44159, 45035, 45036, 45038, 45039]
                        ])
                    else:
                        measure_on_datasets = get_validation_performance(model_sklearn)
                    #except:
                    #    print("Failed to get validation performance")
                    #    measure_on_datasets = {}
                else:
                    measure_on_datasets = {}
                metric_dic = {
                        "epoch": epoch,
                        "train_loss": total_loss,
                        "balanced_acc": balanced_acc_tabpfn,
                        "balanced_acc_lasso": balanced_acc_lasso,
                        "mean_diff_balanced_acc": mean_diff_balanced_acc,
                        "mean_relative_diff_balanced_acc": mean_relative_diff_balanced_acc,
                        "num_features_no_pad": extra_prior_kwargs_dict["hyperparameters"]['num_features_no_pad'],
                        "mean_val_loss": mean_val_loss,
                        "val_loss": val_score,
                        "lr": optimizer.param_groups[0]['lr'],
                        "time_to_get_batch": time_to_get_batch,
                        "forward_time": forward_time,
                        "step_time": step_time,
                        "nan_share": nan_share,
                        "ignore_share": ignore_share,
                        **measure_on_datasets
                    }
                if use_wandb:
                    try:
                        wandb.log(metric_dic)
                    except Exception as e:
                        print("Failed to log to wandb on epoch", epoch)
                        print("Printing exception", e)
                if use_neptune:
                    try:
                        for key, value in metric_dic.items():
                            run["metrics/" + key].append(value)
                    except Exception as e:
                        print("Failed to log to neptune on epoch", epoch)
                        print("Printing exception", e)
                    
            
            if epoch % save_every == 0 and epoch > 0 and rank == 0:
                # Save model
                #TODO save config
                torch.save({"model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "epoch": epoch,
                            "num_features_no_pad": extra_prior_kwargs_dict["hyperparameters"]['num_features_no_pad'],
                            "config":remove_callables(config)}, f"./model_checkpoints/model_{name}_{epoch}.pt")
                print(f"./model_checkpoints/model_{name}_{epoch}.pt")

            if verbose:
                print('-' * 89)
                print(
                    f'| end of epoch {epoch:3d} | time: {(time.time() - epoch_start_time):5.2f}s | mean loss {total_loss:5.2f} | '
                    f"pos losses {','.join([f'{l:5.2f}' for l in total_positional_losses])}, lr {scheduler.get_last_lr()[0]}"
                    f' data time {time_to_get_batch:5.2f} step time {step_time:5.2f}'
                    f' forward time {forward_time:5.2f}' 
                    f' nan share {nan_share:5.2f} ignore share (for classification tasks) {ignore_share:5.4f}'
                    + (f'val score {val_score}' if val_score is not None else ''))
                print('-' * 89)



            # stepping with wallclock time based scheduler
            if epoch_callback is not None and rank == 0:
                epoch_callback(model, epoch / epochs)
            scheduler.step()
    except KeyboardInterrupt:
        pass

    if rank == 0: # trivially true for non-parallel training
        # finish neptune run
        if use_neptune:
            run.stop()
        # finish wandb run
        if use_wandb:
            wandb.finish()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module
            dl = None
        return total_loss, total_positional_losses, model.to('cpu'), dl

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


if __name__ == '__main__':
    config_parser = argparse.ArgumentParser(description='Only used as a first parser for the config file path.')
    config_parser.add_argument('--config')
    parser = argparse.ArgumentParser()
    parser.add_argument('prior')
    parser.add_argument('--loss_function', default='barnll')
    # Optional Arg's for `--loss_function barnll`
    parser.add_argument('--min_y', type=float, help='barnll can only model y in strict ranges, this is the minimum y can take.')
    parser.add_argument('--max_y', type=float, help='barnll can only model y in strict ranges, this is the maximum y can take.')
    parser.add_argument('--num_buckets', default=100, type=int)
    #parser.add_argument('--num_features', default=None, type=int, help='Specify depending on the prior.')
    parser.add_argument("--extra_prior_kwargs_dict", default={}, dest="extra_prior_kwargs_dict", action=StoreDictKeyPair, nargs="+", metavar="KEY=VAL", help='Specify depending on the prior.')
    parser.add_argument('--encoder', default='linear', type=str, help='Specify depending on the prior.')
    parser.add_argument('--y_encoder', default='linear', type=str, help='Specify depending on the prior. You should specify this if you do not fuse x and y.')
    parser.add_argument('--pos_encoder', default='none', type=str, help='Specify depending on the prior.')
    parser.add_argument('--bptt', default=10, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--warmup_epochs', default=50, type=int)
    parser.add_argument('--validation_period', default=10, type=int)
    parser.add_argument('--permutation_invariant_max_eval_pos', default=None, type=int, help='Set this to an int to ')
    parser.add_argument('--permutation_invariant_sampling', default='weighted', help="Only relevant if --permutation_invariant_max_eval_pos is set.")
    parser.add_argument('--train_mixed_precision', action='store_true')

    # these can likely be mostly left at defaults
    parser.add_argument('--emsize', default=512, type=int) # sometimes even larger is better e.g. 1024
    parser.add_argument('--nlayers', default=6, type=int)
    parser.add_argument('--nhid', default=None, type=int) # 2*emsize is the default
    parser.add_argument('--nhead', default=4, type=int) # nhead = emsize / 64 in the original paper
    parser.add_argument('--dropout', default=.0, type=float)
    parser.add_argument('--steps_per_epoch', default=10, type=int)
    parser.add_argument('--batch_size', default=1000, type=int)
    parser.add_argument('--lr', '--learning_rate', default=.001, type=float) # try also .0003, .0001, go lower with lower batch size

    args, _ = _parse_args(config_parser, parser)

    if args.nhid is None:
        args.nhid = 2*args.emsize

    prior = args.__dict__.pop('prior')

    if prior == 'gp':
        prior = priors.fast_gp.DataLoader
    elif prior == 'ridge':
        prior = priors.ridge.DataLoader
    elif prior == 'stroke':
        prior = priors.stroke.DataLoader
    elif prior == 'mix_gp':
        prior = priors.fast_gp_mix.DataLoader
    else:
        raise NotImplementedError(f'Prior == {prior}.')

    loss_function = args.__dict__.pop('loss_function')

    criterion = nn.GaussianNLLLoss(reduction='none', full=True)
    classificiation_criterion = nn.CrossEntropyLoss(reduction='none')
    num_buckets = args.__dict__.pop('num_buckets')
    max_y = args.__dict__.pop('max_y')
    min_y = args.__dict__.pop('min_y')
    # criterion = nn.MSELoss(reduction='none')

    if loss_function == 'ce':
        criterion = nn.CrossEntropyLoss(reduction='none')
    elif loss_function == 'gaussnll':
        criterion = nn.GaussianNLLLoss(reduction='none', full=True)
    elif loss_function == 'mse':
        criterion = nn.MSELoss(reduction='none')
    else:
        raise NotImplementedError(f'loss_function == {loss_function}.')



    encoder = args.__dict__.pop('encoder')
    y_encoder = args.__dict__.pop('y_encoder')

    def get_encoder_generator(encoder):
        if encoder == 'linear':
            encoder_generator = encoders.Linear
        elif encoder == 'mlp':
            encoder_generator = encoders.MLP
        elif encoder == 'positional':
            encoder_generator = encoders.Positional
        else:
            raise NotImplementedError(f'A {encoder} encoder is not valid.')
        return encoder_generator

    encoder_generator = get_encoder_generator(encoder)
    y_encoder_generator = get_encoder_generator(y_encoder)

    pos_encoder = args.__dict__.pop('pos_encoder')

    if pos_encoder == 'none':
        pos_encoder_generator = None
    elif pos_encoder == 'sinus':
        pos_encoder_generator = positional_encodings.PositionalEncoding
    elif pos_encoder == 'learned':
        pos_encoder_generator = positional_encodings.LearnedPositionalEncoding
    elif pos_encoder == 'paired_scrambled_learned':
        pos_encoder_generator = positional_encodings.PairedScrambledPositionalEncodings
    else:
        raise NotImplementedError(f'pos_encoer == {pos_encoder} is not valid.')

    permutation_invariant_max_eval_pos = args.__dict__.pop('permutation_invariant_max_eval_pos')
    permutation_invariant_sampling = args.__dict__.pop('permutation_invariant_sampling')
    if permutation_invariant_max_eval_pos is not None:
        if permutation_invariant_sampling == 'weighted':
            get_sampler = get_weighted_single_eval_pos_sampler
        elif permutation_invariant_sampling == 'uniform':
            get_sampler = get_uniform_single_eval_pos_sampler
        else:
            raise ValueError()
        args.__dict__['single_eval_pos_gen'] = get_sampler(permutation_invariant_max_eval_pos)


    print("ARGS for `train`:", args.__dict__)

    train(prior, criterion, encoder_generator,
          y_encoder_generator=y_encoder_generator, pos_encoder_generator=pos_encoder_generator,
          **args.__dict__)

