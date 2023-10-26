import random
import math

import torch
from torch import nn
import numpy as np

from tabpfn.utils import default_device
from .utils import get_batch_to_dataloader


import random
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import time
from joblib import Parallel, delayed
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from tabpfn.utils import normalize_by_used_features_f

import time
import random

import torch
from torch import nn

from .utils import get_batch_to_dataloader
from tabpfn.utils import normalize_data, nan_handling_missing_for_unknown_reason_value, nan_handling_missing_for_no_reason_value, nan_handling_missing_for_a_reason_value, to_ranking_low_mem, remove_outliers, normalize_by_used_features_f
from .utils import randomize_classes, CategoricalActivation
from .utils import uniform_int_sampler_f


def randomize_leaves_func(forest, n_classes):
    class_labels_used = np.zeros(n_classes, dtype=bool)
    for tree in forest.estimators_:
        n_nodes = tree.tree_.node_count
        is_leaf = np.zeros(shape=n_nodes, dtype=bool)
    
        for i in range(n_nodes):
            if tree.tree_.children_left[i] == -1:
                is_leaf[i] = True

        leaf_indices = np.where(is_leaf)[0]

        # Generate random class labels for each leaf node
        random_class_labels = np.random.randint(0, n_classes, len(leaf_indices))
        for label in random_class_labels:
            if not class_labels_used[label]:
                class_labels_used[label] = True

        # Replace leaf node values with random class labels
        for i, leaf_idx in enumerate(leaf_indices):
            tree.tree_.value[leaf_idx] = np.zeros((1, n_classes))
            tree.tree_.value[leaf_idx, 0, random_class_labels[i]] = 1
    
    # check that all classes have been used
    #TODO
    # if not np.all(class_labels_used):
    #     print("Warning: not all classes have been used")
    #     print("Trying again...")
    #     return randomize_leaves_func(forest, n_classes)
    return forest


@torch.no_grad()
def get_batch_trees(batch_size, seq_len, num_features_max, num_features_sampler, hyperparameters, device=default_device, num_outputs=1, sampling='normal'
              , epoch=None, **kwargs):
    if 'multiclass_type' in hyperparameters and hyperparameters['multiclass_type'] == 'multi_node':
        num_outputs = num_outputs * hyperparameters['num_classes']

    correlation_strength = np.random.uniform(hyperparameters['correlation_strength_min'], hyperparameters['correlation_strength_max'])
    correlation_proba = np.random.uniform(hyperparameters['correlation_proba_min'], hyperparameters['correlation_proba_max'])
    def get_seq(num_features=None):
        if num_features is None:
            num_features = num_features_sampler()
        if sampling == 'normal':
            # generate a random covariance matrix
            cov = np.eye(num_features)
            for i in range(num_features):
                for j in range(i + 1, num_features):
                    if np.random.random() < correlation_proba:
                        cov[i, j] = cov[j, i] = np.random.normal(0, correlation_strength)
            # Make sure the covariance matrix is positive definite
            cov = np.dot(cov, cov.T)
            # generate a mutlivariate normal distribution
            data = np.random.multivariate_normal(np.zeros(num_features), cov, seq_len).astype(np.float32).reshape(seq_len, 1, num_features)
        elif sampling == 'mixed':
            zipf_p, multi_p, normal_p = random.random() * 0.66, random.random() * 0.66, random.random() * 0.66
            def sample_data(n):
                if random.random() > normal_p:
                    #TODO check pre-sample causes
                    return torch.normal(0., 1., (seq_len, 1), device="cpu").float()
                elif random.random() > multi_p:
                    x = torch.multinomial(torch.rand((random.randint(2, 10))), seq_len, replacement=True).unsqueeze(-1).float()
                    x = (x - torch.mean(x)) / torch.std(x)
                    return x
                else:
                    x = torch.minimum(torch.tensor(np.random.zipf(2.0 + random.random() * 2, size=(seq_len)),
                                        device="cpu").unsqueeze(-1).float(), torch.tensor(10.0, device="cpu"))
                    return x - torch.mean(x)
            data = torch.cat([sample_data(n).unsqueeze(-1) for n in range(num_features)], -1)
            # create random correlations
            for i in range(num_features):
                for j in range(num_features):
                    if i != j and random.random() > correlation_proba:
                        data[:, :, i] += random.random() * data[:, :, j] * correlation_strength
        elif sampling == 'uniform':
            data = torch.rand((seq_len, 1, num_features), device="cpu")
        else:
            raise ValueError(f'Sampling is set to invalid setting: {sampling}.')
        # Select categorical features
        data = data.reshape(-1, num_features)
        n_categorical_features = np.random.binomial(data.shape[1], hyperparameters['p_categorical'])
        categorical_features = np.random.choice(data.shape[1], n_categorical_features, replace=False)
        # Convert numerical to categorical
        for i in categorical_features:
            n_categories = np.random.randint(hyperparameters['min_categories'], hyperparameters['max_categories'])
            #TODO check if this is the right distribution
            probas = np.random.dirichlet(np.ones(n_categories))
            # Convert to categorical
            data[:, i] = np.random.choice(n_categories, data.shape[0], p=probas)

        # Remove random features
        # X will be used to fit the forest, but `data` will be returned
        X = data.reshape(-1, num_features)
        if hyperparameters.get("random_feature_removal", 0) > 0:
            # sample the number of features to remove
            number_of_features_to_remove = int(np.random.uniform(hyperparameters["random_feature_removal_min"], hyperparameters["random_feature_removal"] * X.shape[1]))
            number_of_features_to_remove = min(number_of_features_to_remove, X.shape[1] -3) # make sure we have at least 3 features
            features_to_remove = np.random.choice(X.shape[1], number_of_features_to_remove, replace=False)
            X = np.delete(X, features_to_remove, axis=1)
            # fix the categorical features
            categorical_features = [i for i in categorical_features if i not in features_to_remove]
            #TODO seems correct, but check
            categorical_features = [i - sum([f < i for f in features_to_remove]) for i in categorical_features]
            
        # One-hot encode categorical features
        if len(categorical_features) > 0:
            cat_encoder = OneHotEncoder(sparse_output=False, categories='auto')
            full_encoder = ColumnTransformer([('cat', cat_encoder, categorical_features)], remainder='passthrough')
            X = full_encoder.fit_transform(X)
        
        # Sample max depth from uniform distribution
        max_depth = 2 + int(np.random.exponential( 1./ hyperparameters['max_depth_lambda']))
        # Sample number of trees from an exponential distribution with parameter lambda
        n_estimators = (1 + int(np.random.exponential(1. / hyperparameters['n_estimators_lambda']))) if (hyperparameters['n_estimators'] is None) else hyperparameters['n_estimators']
        time_forest = time.time()
        forest = ExtraTreesClassifier(max_depth=max_depth, n_estimators=n_estimators, max_features=1, n_jobs=1) # max_features=1 means that the splits are totally random
        if "num_classes_tree" in hyperparameters and hyperparameters["num_classes_tree"] is not None:
            num_classes = hyperparameters["num_classes_tree"]
        else:
            num_classes = hyperparameters['num_classes']  

        fake_y = np.random.randint(0, num_classes, data.shape[0])
        # label encoder to prevent holes in the class labels (e.g. 0, 1, 3)
        le = LabelEncoder()
        fake_y = le.fit_transform(fake_y)
        forest.fit(X,fake_y) #TODO unbalance
        
        if hyperparameters["randomize_leaves"]:
            forest = randomize_leaves_func(forest, num_classes)
        if hyperparameters["return_classes_in_trees_prior"]:
            y = forest.predict(X)
        else:
            # uses flexible_categorical class_assigner
            # select a class at random
            #predictions = forest.predict_proba(X)
            #i = np.random.randint(0, predictions.shape[1])
            i = 0
            y = forest.predict_proba(X)[:, i]
        time_forest = time.time() - time_forest
        if np.random.random() < 0.001:
            print("Time to fit forest: ", time_forest)

        data = torch.from_numpy(data)#.to(device)
        # random feature rotation
        if hyperparameters.get("random_feature_rotation", False):
            data = data[..., (torch.arange(data.shape[-1], device="cpu")+random.randrange(data.shape[-1])) % data.shape[-1]]
        
        # # normalize by used features
        # if hyperparameters["normalize_by_used_features"]:
        #     print("Normalizing by used features")
        #     print("data.shape", data.shape)
        #     print("num_features_max", num_features_max)
        #     data = normalize_by_used_features_f(data, data.shape[-1], num_features_max, normalize_with_sqrt=hyperparameters["normalize_with_sqrt"])

        return data.reshape(-1, 1, num_features).float(), torch.from_numpy(y).reshape(-1, 1, num_outputs).float()

    

    # if hyperparameters.get('new_forest_per_example', False):
    #     get_model = lambda: generate_random_forest(hyperparameters)
    # else:
    #     model = generate_random_forest(hyperparameters)
    #     get_model = lambda: model
    if hyperparameters['num_features_fixed']:
        num_features = num_features_sampler()
    else:
        num_features = None
    sample = [get_seq(num_features) for _ in range(0, batch_size)]

    x, y = zip(*sample)
    y = torch.cat(y, 1).detach().squeeze(2)
    # concat x and pad with zeros
    # first pad with zeros to num_features_max
    # we pad here so we can have different number of features in the same batch
    n_features_used = [x[i].shape[-1] for i in range(len(x))]
    x = [torch.cat([x[i], torch.zeros((seq_len, 1, num_features_max - x[i].shape[-1]))], -1) for i in range(len(x))]
    # then concat
    x = torch.cat(x, 1).detach()
    
    return x, y, n_features_used  # x.shape = (T,B,H)


time_it = False
from priors.flexible_categorical import BalancedBinarize, RegressionNormalized, MulticlassRank, MulticlassValue, MulticlassMultiNode

    
def normalize_by_used_features_variable_n_features_f(x, n_features_used, num_features_max, normalize_with_sqrt=False):
    # n_features_used has shape (B,)
    assert len(n_features_used) == x.shape[1]
    # copy x
    x = x.clone()
    if normalize_with_sqrt:
        for b in range(x.shape[1]):
            x[:, b, :] = x[:, b, :] / (n_features_used[b] / num_features_max)**(1 / 2)
    else:
        for b in range(x.shape[1]):
            x[:, b, :] = x[:, b, :] / (n_features_used[b] / num_features_max)
    return x


class FlexibleCategorical(torch.nn.Module):
    def __init__(self, get_batch, hyperparameters, args):
        super(FlexibleCategorical, self).__init__()

        self.h = {k: hyperparameters[k]() if callable(hyperparameters[k]) else hyperparameters[k] for k in
                                hyperparameters.keys()}
        self.args = args
        self.args_passed = {**self.args}
        self.args_passed.update({'num_features': self.h['num_features_used']})
        self.get_batch = get_batch


        if self.h['num_classes'] == 0:
            self.class_assigner = RegressionNormalized()
        else:
            if self.h['num_classes'] > 1 and not self.h['balanced']:
                if self.h['multiclass_type'] == 'rank':
                    self.class_assigner = MulticlassRank(self.h['num_classes']
                                                 , ordered_p=self.h['output_multiclass_ordered_p']
                                                 )
                elif self.h['multiclass_type'] == 'value':
                    self.class_assigner = MulticlassValue(self.h['num_classes']
                                                         , ordered_p=self.h['output_multiclass_ordered_p']
                                                         )
                elif self.h['multiclass_type'] == 'multi_node':
                    self.class_assigner = MulticlassMultiNode(self.h['num_classes'])
                else:
                    raise ValueError("Unknow Multiclass type")
            elif self.h['num_classes'] == 2 and self.h['balanced']:
                self.class_assigner = BalancedBinarize()
            elif self.h['num_classes'] > 2 and self.h['balanced']:
                raise NotImplementedError("Balanced multiclass training is not possible")

    def drop_for_reason(self, x, v):
        nan_prob_sampler = CategoricalActivation(ordered_p=0.0
                                                 , categorical_p=1.0
                                                 , keep_activation_size=False,
                                                 num_classes_sampler=lambda: 20)
        d = nan_prob_sampler(x)
        # TODO: Make a different ordering for each activation
        x[d < torch.rand((1,), device=x.device) * 20 * self.h['nan_prob_no_reason'] * random.random()] = v
        return x

    def drop_for_no_reason(self, x, v):
        x[torch.rand(x.shape, device=self.args['device']) < random.random() * self.h['nan_prob_no_reason']] = v
        return x

    def forward(self, batch_size):
        start = time.time()
        x, y, n_features_used = self.get_batch(hyperparameters=self.h, **self.args_passed)
        if time_it:
            print('Flex Forward Block 1', round(time.time() - start, 3))

        start = time.time()

        if self.h['nan_prob_no_reason']+self.h['nan_prob_a_reason']+self.h['nan_prob_unknown_reason'] > 0 and random.random() > 0.5: # Only one out of two datasets should have nans
            if random.random() < self.h['nan_prob_no_reason']: # Missing for no reason
                x = self.drop_for_no_reason(x, nan_handling_missing_for_no_reason_value(self.h['set_value_to_nan']))

            if self.h['nan_prob_a_reason'] > 0 and random.random() > 0.5: # Missing for a reason
                x = self.drop_for_reason(x, nan_handling_missing_for_a_reason_value(self.h['set_value_to_nan']))

            if self.h['nan_prob_unknown_reason'] > 0: # Missing for unknown reason  and random.random() > 0.5
                if random.random() < self.h['nan_prob_unknown_reason_reason_prior']:
                    x = self.drop_for_no_reason(x, nan_handling_missing_for_unknown_reason_value(self.h['set_value_to_nan']))
                else:
                    x = self.drop_for_reason(x, nan_handling_missing_for_unknown_reason_value(self.h['set_value_to_nan']))

        # Categorical features
        if 'categorical_feature_p' in self.h and random.random() < self.h['categorical_feature_p']:
            p = random.random()
            for col in range(x.shape[2]):
                num_unique_features = max(round(random.gammavariate(1,10)),2)
                m = MulticlassRank(num_unique_features, ordered_p=0.3)
                if random.random() < p:
                    x[:, :, col] = m(x[:, :, col])

        if time_it:
            print('Flex Forward Block 2', round(time.time() - start, 3))
            start = time.time()

        if self.h['normalize_to_ranking']:
            x = to_ranking_low_mem(x)
        else:
            x = remove_outliers(x)

        # checks for class assignment before normalization
        if not self.h.get('return_classes_in_trees_prior', False):
            # check that the class has not been assigned yet
            assert not (y.long() == y).all(), f"Classes already assigned: {y[:10]}"
        else:
            # check that the class has been assigned
            assert (y.long() == y).all(), f"Classes not assigned: {y[:10]}"

        x, y = normalize_data(x), normalize_data(y)

        if time_it:
            print('Flex Forward Block 3', round(time.time() - start, 3))
            start = time.time()

        # Cast to classification if enabled
        if not self.h.get('return_classes_in_trees_prior', False):
            y = self.class_assigner(y).float()

        if time_it:
            print('Flex Forward Block 4', round(time.time() - start, 3))
            start = time.time()
        if self.h['normalize_by_used_features']:
                #x = normalize_by_used_features_f(x, self.h['num_features_used'], self.args['num_features'], normalize_with_sqrt=self.h.get('normalize_with_sqrt',False))
                x = normalize_by_used_features_variable_n_features_f(x, n_features_used, self.args['num_features'], normalize_with_sqrt=self.h.get('normalize_with_sqrt',False))
        if time_it:
            print('Flex Forward Block 5', round(time.time() - start, 3))

        start = time.time()
        # Done in the tree prior
        # Append empty features if enabled
        # x = torch.cat(
        #     [x, torch.zeros((x.shape[0], x.shape[1], self.args['num_features'] - self.h['num_features_used']),
        #                     device=x.device)], -1)
        if time_it:
            print('Flex Forward Block 6', round(time.time() - start, 3))

        if torch.isnan(y).sum() > 0:
            print('Nans in target!')

        if self.h['check_is_compatible']:
            for b in range(y.shape[1]):
                is_compatible, N = False, 0
                while not is_compatible and N < 10:
                    targets_in_train = torch.unique(y[:self.args['single_eval_pos'], b], sorted=True)
                    targets_in_eval = torch.unique(y[self.args['single_eval_pos']:, b], sorted=True)

                    is_compatible = len(targets_in_train) == len(targets_in_eval) and (
                                targets_in_train == targets_in_eval).all() and len(targets_in_train) > 1

                    if not is_compatible:
                        randperm = torch.randperm(x.shape[0])
                        x[:, b], y[:, b] = x[randperm, b], y[randperm, b]
                    N = N + 1
                if not is_compatible:
                    if not is_compatible:
                        # todo check that it really does this and how many together
                        y[:, b] = -100 # Relies on CE having `ignore_index` set to -100 (default)

        if self.h['normalize_labels']:
            #assert self.h['output_multiclass_ordered_p'] == 0., "normalize_labels destroys ordering of labels anyways."
            for b in range(y.shape[1]):
                valid_labels = y[:,b] != -100
                if self.h.get('normalize_ignore_label_too', False):
                    valid_labels[:] = True
                y[valid_labels, b] = (y[valid_labels, b] > y[valid_labels, b].unique().unsqueeze(1)).sum(axis=0).unsqueeze(0).float()

                if y[valid_labels, b].numel() != 0 and self.h.get('rotate_normalized_labels', True):
                    num_classes_float = (y[valid_labels, b].max() + 1).cpu()
                    num_classes = num_classes_float.int().item()
                    assert num_classes == num_classes_float.item()
                    random_shift = torch.randint(0, num_classes, (1,), device=y.device)#device=self.args['device'])
                    y[valid_labels, b] = (y[valid_labels, b] + random_shift) % num_classes

        return x, y, y  # x.shape = (T,B,H)


@torch.no_grad()
def get_batch(batch_size, seq_len, num_features, device, hyperparameters=None, batch_size_per_gp_sample=None, **kwargs):
    batch_size_per_gp_sample = batch_size_per_gp_sample or (min(32, batch_size))
    num_models = batch_size // batch_size_per_gp_sample
    print("num models", num_models)
    assert num_models > 0, f'Batch size ({batch_size}) is too small for batch_size_per_gp_sample ({batch_size_per_gp_sample})'
    assert num_models * batch_size_per_gp_sample == batch_size, f'Batch size ({batch_size}) not divisible by batch_size_per_gp_sample ({batch_size_per_gp_sample})'

    # Sample one seq_len for entire batch
    seq_len = hyperparameters['seq_len_used']() if callable(hyperparameters['seq_len_used']) else seq_len
    print("seq_len", seq_len)

    args = {'device': device, 'seq_len': seq_len, 'num_features': num_features, 'batch_size': batch_size_per_gp_sample, **kwargs}

    models = [FlexibleCategorical(get_batch_trees, hyperparameters, args) for _ in range(num_models)]

    sample = [model(batch_size=batch_size_per_gp_sample) for model in models]

    x, y, y_ = zip(*sample)
    x, y, y_ = torch.cat(x, 1).detach(), torch.cat(y, 1).detach(), torch.cat(y_, 1).detach()

    return x, y, y_





DataLoader = get_batch_to_dataloader(get_batch)
