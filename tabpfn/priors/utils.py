import random

import torch

from tabpfn.utils import set_locals_in_self
from .prior import PriorDataLoader
from torch import nn
import numpy as np
import scipy.stats as stats
import math

from torch.utils.data import IterableDataset, DataLoader
from typing import Callable, Optional
import math
import torch
import numpy as np
import random


def make_dataloader(dataloader_kwargs, get_batch_kwargs, test_loader=False):
    if test_loader:
        get_batch_kwargs["batch_size"] = 1
        dataloader_kwargs["num_workers"] = 0
        dataloader_kwargs["pin_memory"] = False
        dataloader_kwargs.pop("prefetch_factor", None)  # Remove key if it exists
        dataloader_kwargs.pop("persistent_workers", None)  # Remove key if it exists

    ds = PriorDataset(**get_batch_kwargs)
    dl = DataLoader(
        ds,
        batch_size=None,
        batch_sampler=None,  # This disables automatic batching
        #test_loader=test_loader,
        **dataloader_kwargs,
    )
    return dl

def get_batch_to_dataloader(get_batch_method):
    def return_dl(dataload_kwargs, get_batch_kwargs, test_loader=False):
        get_batch_kwargs["get_batch_method"] = get_batch_method
        return make_dataloader(dataload_kwargs, get_batch_kwargs, test_loader=test_loader)
    return return_dl


class PriorDataset(IterableDataset):
    def __init__(
        self,
        num_steps: int,
        batch_size: int,
        eval_pos_seq_len_sampler: Callable,
        get_batch_method: Callable,
        num_features,
        seq_len_maximum: Optional[int] = None,
        device: Optional[str] = "cpu",
        test_loader: Optional[bool] = False,
        **get_batch_kwargs,
    ):
        # The stuff outside the or is set as class attribute before instantiation.
        self.num_features = num_features
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.eval_pos_seq_len_sampler = eval_pos_seq_len_sampler
        self.seq_len_maximum = seq_len_maximum
        self.device = device
        self.get_batch_kwargs = get_batch_kwargs
        self.get_batch_method = get_batch_method
        self.model = None
        self.epoch = 0
        self.step = 0
        self.test_loader = test_loader
        print("DataLoader.__dict__", self.__dict__)

    def gbm(self):
        single_eval_pos, seq_len = self.eval_pos_seq_len_sampler()
        # Scales the batch size dynamically with the power of 'dynamic_batch_size'.
        # A transformer with quadratic memory usage in the seq len would need a power of 2 to keep memory constant.
        # if 'dynamic_batch_size' in kwargs and kwargs['dynamic_batch_size'] > 0 and kwargs[
        #    'dynamic_batch_size'] is not None:
        #    kwargs['batch_size'] = kwargs['batch_size'] * math.floor(
        #        math.pow(kwargs['seq_len_maximum'], kwargs['dynamic_batch_size'])
        #        / math.pow(kwargs['seq_len'], kwargs['dynamic_batch_size'])
        #    )
        batch = self.get_batch_method(
            single_eval_pos=single_eval_pos,
            seq_len=seq_len,
            batch_size=self.batch_size,
            num_features=self.num_features,
            device=self.device,
            model=self.model,
            epoch=self.epoch,
            test_batch=self.test_loader,
            **self.get_batch_kwargs,
        )
        #TODO
        #if batch.single_eval_pos is None:
        #    batch.single_eval_pos = single_eval_pos
        self.step += self.batch_size
        if self.step >= self.num_steps:
            self.step -= self.num_steps
            self.epoch += 1
        print("Num steps", self.num_steps, "Step", self.step)
        

        return (0, batch[0], batch[1]), batch[2], single_eval_pos # data (style, x, y), target, single_eval_pos

    def __len__(self):
        return self.num_steps

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        # Different workers should have different seeds for numpy, python (pytorch automatic)
        num_steps = self.num_steps
        if worker_info is not None:
            np.random.seed(worker_info.seed % (2**32))
            random.seed(worker_info.seed % (2**32))
            num_steps = math.ceil(
                self.num_steps / worker_info.num_workers
            )  # Rounding up means some workers will do unnecessary work, but that's fine.

        # TODO: Why do we assign model, do we want to keep that behavior?
        # assert hasattr(self, 'model'), "Please assign model with `dl.model = ...` before training."
        if num_steps > 0:
            it = iter(self.gbm() for _ in range(num_steps))
        else:
            it = iter(self.gbm, 1)
        return it
    
def plot_features(data, targets, fig=None, categorical=True):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()
    #data = np.concatenate([data, np.expand_dims(targets, -1)], -1)
    #df = pd.DataFrame(data, columns=list(range(0, data.shape[1])))
    #g = sns.pairplot(df, hue=data.shape[1]-1, palette="Set2", diag_kind="kde", height=2.5)
    #plt.legend([], [], frameon=False)
    #g._legend.remove()
    #g = sns.PairGrid(df, hue=data.shape[1]-1)
    #g.map_diag(sns.histplot)
    #g.map_offdiag(sns.scatterplot)
    #g._legend.remove()

    fig2 = fig if fig else plt.figure(figsize=(8, 8))
    spec2 = gridspec.GridSpec(ncols=data.shape[1], nrows=data.shape[1], figure=fig2)
    for d in range(0, data.shape[1]):
        for d2 in range(0, data.shape[1]):
            if d > d2:
                continue
            sub_ax = fig2.add_subplot(spec2[d, d2])
            sub_ax.set_xticks([])
            sub_ax.set_yticks([])
            if d == d2:
                if categorical:
                    sns.kdeplot(x=data[:, d],hue=targets[:],ax=sub_ax,legend=False, palette="deep")
                else:
                    sns.kdeplot(x=data[:, d], ax=sub_ax, legend=False)
                sub_ax.set(ylabel=None)
            else:
                if categorical:
                    sns.scatterplot(x=data[:, d], y=data[:, d2],
                           hue=targets[:],legend=False, palette="deep")
                else:
                    sns.scatterplot(x=data[:, d], y=data[:, d2],
                                    hue=targets[:], legend=False)
                #plt.scatter(data[:, d], data[:, d2],
                #               c=targets[:])
            #sub_ax.get_xaxis().set_ticks([])
            #sub_ax.get_yaxis().set_ticks([])
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    fig2.show()


def plot_prior(prior):
    import matplotlib.pyplot as plt
    s = np.array([prior() for _ in range(0, 1000)])
    count, bins, ignored = plt.hist(s, 50, density=True)
    print(s.min())
    plt.show()

trunc_norm_sampler_f = lambda mu, sigma : lambda: stats.truncnorm((0 - mu) / sigma, (1000000 - mu) / sigma, loc=mu, scale=sigma).rvs(1)[0]
beta_sampler_f = lambda a, b : lambda : np.random.beta(a, b)
gamma_sampler_f = lambda a, b : lambda : np.random.gamma(a, b)
uniform_sampler_f = lambda a, b : lambda : np.random.uniform(a, b)
uniform_int_sampler_f = lambda a, b : lambda : round(np.random.uniform(a, b))
def zipf_sampler_f(a, b, c):
    x = np.arange(b, c)
    weights = x ** (-a)
    weights /= weights.sum()
    return lambda : stats.rv_discrete(name='bounded_zipf', values=(x, weights)).rvs(1)
scaled_beta_sampler_f = lambda a, b, scale, minimum : lambda : minimum + round(beta_sampler_f(a, b)() * (scale - minimum))

def order_by_y(x, y):
    order = torch.argsort(y if random.randint(0, 1) else -y, dim=0)[:, 0, 0]
    order = order.reshape(2, -1).transpose(0, 1).reshape(-1)#.reshape(seq_len)
    x = x[order]  # .reshape(2, -1).transpose(0, 1).reshape(-1).flip([0]).reshape(seq_len, 1, -1)
    y = y[order]  # .reshape(2, -1).transpose(0, 1).reshape(-1).reshape(seq_len, 1, -1)

    return x, y

def randomize_classes(x, num_classes):
    classes = torch.arange(0, num_classes, device=x.device)
    random_classes = torch.randperm(num_classes, device=x.device).type(x.type())
    x = ((x.unsqueeze(-1) == classes) * random_classes).sum(-1)
    return x


class CategoricalActivation(nn.Module):
    def __init__(self, categorical_p=0.1, ordered_p=0.7
                 , keep_activation_size=False
                 , num_classes_sampler=zipf_sampler_f(0.8, 1, 10)):
        self.categorical_p = categorical_p
        self.ordered_p = ordered_p
        self.keep_activation_size = keep_activation_size
        self.num_classes_sampler = num_classes_sampler

        super().__init__()

    def forward(self, x):
        # x shape: T, B, H

        x = nn.Softsign()(x)

        num_classes = self.num_classes_sampler()
        hid_strength = torch.abs(x).mean(0).unsqueeze(0) if self.keep_activation_size else None

        categorical_classes = torch.rand((x.shape[1], x.shape[2])) < self.categorical_p
        class_boundaries = torch.zeros((num_classes - 1, x.shape[1], x.shape[2]), device=x.device, dtype=x.dtype)
        # Sample a different index for each hidden dimension, but shared for all batches
        for b in range(x.shape[1]):
            for h in range(x.shape[2]):
                ind = torch.randint(0, x.shape[0], (num_classes - 1,))
                class_boundaries[:, b, h] = x[ind, b, h]

        for b in range(x.shape[1]):
            x_rel = x[:, b, categorical_classes[b]]
            boundaries_rel = class_boundaries[:, b, categorical_classes[b]].unsqueeze(1)
            x[:, b, categorical_classes[b]] = (x_rel > boundaries_rel).sum(dim=0).float() - num_classes / 2

        ordered_classes = torch.rand((x.shape[1],x.shape[2])) < self.ordered_p
        ordered_classes = torch.logical_and(ordered_classes, categorical_classes)
        x[:, ordered_classes] = randomize_classes(x[:, ordered_classes], num_classes)

        x = x * hid_strength if self.keep_activation_size else x

        return x
