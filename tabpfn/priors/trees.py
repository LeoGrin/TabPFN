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

# class Tree:
#     def __init__(self, depth, parent=None):
#         self.parent = parent
#         self.left = None
#         self.right = None
#         self.root_feature = None
#         self.root_threshold = None
#         self.depth = depth
#         self.predicion = None

#     def set_split(self, feature_id, threshold):
#         self.root_threshold = threshold
#         self.root_feature = feature_id

#     def set_prediction(self, prediction):
#         assert self.left is None and self.right is None  # leaf
#         self.predicion = prediction

#     def predict(self, x):
#         #x shape (n_features,)
#         if not self.predicion is None:
#             return self.predicion
#         else:
#             if x[self.root_feature] > self.root_threshold:
#                 return self.right.predict(x)
#             else:
#                 return self.left.predict(x)


# class Forest:
#     def __init__(self, tree_list, rng):
#         self.tree_list = tree_list
#         self.rng = rng
#     def predict(self, x):
#         #x shape (n_samples, n_features)

#         # for tree in self.tree_list:
#         #     
#         # ("depth: {}".format(tree.depth))
#         #     
#         # ("split {}".format(tree.root_threshold))#
#         #
#         #     sum = 0
#         #     for sample in x:
#         #         sum += tree.predict(sample)
#         #     
#         # (sum)
#         predictions = []
#         for sample in x: #TODO vectorize ?
#             values, counts = np.unique([tree.predict(sample) for tree in self.tree_list], return_counts=True)
#             indices_max = np.argwhere(counts == np.amax(counts)).flatten()
#             prediction = self.rng.choice(values[indices_max], 1)[0]
#             predictions.append(prediction)
#         return np.array(predictions)

# def generate_random_tree(x, n_classes, depth, split_distribution="uniform", split_param=1, rng=None):
#     """
#     Generate a random tree which labels the data.
#     :param x: data to label
#     :param n_classes: number of classes to choose for the labels
#     :param depth: depth of the random tree
#     :param split_distribution:{"uniform", "gaussian"}, default="uniform"
#      Distribution from which is sampled the split threshold at each step.
#      If "uniform": Uniform(25% quantile, 75% quantile)
#      If "gaussian": N(0, split_param * interquartile range)
#     :param split_param: parameter controlling the spread of the split threshold distribution
#     WARNING: not implement yet for "uniform" split distribution
#     :return: a Tree object with a fit and predict methods
#     """
#     if rng is None:
#         rng = np.random.RandomState()
#     def generate_tree(x, n_classes, depth, parent=None, prediction=None, min_num_leaf=5):
#         #TODO allow random stopping of a substree?
#         if x.shape[0] < min_num_leaf or depth == 1:
#             #TODO: when stopping early, correct the depth of all parents
#             #you should look at all leaf of a tree for this
#             assert depth > 1 or prediction is not None
#             if depth > 1:
#                 prediction = rng.choice(range(n_classes))
#             leaf = Tree(depth, parent)
#             leaf.set_prediction(prediction)
#             return leaf
#         else:
#             x_median = np.quantile(x, 0.5, axis=0)
#             x_25 = np.quantile(x, 0.25, axis=0)
#             x_75 = np.quantile(x, 0.75, axis=0)
#             n_features = x.shape[1]

#             tree = Tree(depth, parent)
#             split_feature = rng.choice(range(n_features), 1)[0]
#             # we want to sample a split threshold depending on the variance of this feature in our data
#             if split_distribution == "uniform":
#                 #TODO allow split_param
#                 split_threshold = rng.uniform(x_25[split_feature],
#                                                     x_75[split_feature])
#             if split_distribution == "gaussian":
#                 split_threshold = rng.normal(loc=x_median[split_feature], scale=split_param * (x_75[split_feature] - x_25[split_feature]))

#             tree.set_split(split_feature, split_threshold)
#             if depth == 2: #make sure two adjacent leaves have different predictions
#                 prediction_left = rng.choice(range(n_classes))
#                 prediction_right = (prediction_left + 1) % 2
#             else:
#                 prediction_left = None
#                 prediction_right = None
#             tree.right = generate_tree(x[x[:, split_feature] >= split_threshold],
#                                        n_classes,
#                                        depth - 1,
#                                        tree, prediction_left)
#             tree.left = generate_tree(x[x[:, split_feature] < split_threshold],
#                                       n_classes,
#                                       depth - 1,
#                                       tree,
#                                       prediction_right)
#             return tree

#     root = generate_tree(x, n_classes, depth, parent=None)
#     return root




    


# def generate_random_forest(x, n_classes=2, n_trees=5, max_depth=5, depth_distribution="constant",
#                                        split_distribution="uniform", split_param=1, rng=None, **kwargs):
#     """
#     :param x: data to label
#     :param n_classes: number of classes to choose for the labels
#     :param n_trees: number of trees in the forest
#     :param max_depth: depth of the random tree
#     :param depth_distribution:{"constant", "uniform"} Distribution from which are sampled the tree depths
#     if "constant", every depths are max_depth
#     if "uniform", uniform in [2, ..., max_depth]
#     :param split_distribution:{"uniform", "gaussian"}, default="uniform"
#      Distribution from which is sampled the split threshold at each step.
#      If "uniform": Uniform(-split_param * feature_std, +split_param * feature_std)
#      If "gaussian": N(0, split_param * feature_std)
#     :param split_param: parameter controlling the spread of the split threshold distribution
#     :return: a Forest object with predict and fit methods
#     """
#     if rng is None:
#         rng = np.random.RandomState()
#     if depth_distribution == "constant":
#         depths = [max_depth] * n_trees
#     elif depth_distribution == "uniform":
#         depths = [rng.randint(2, max_depth + 1) for i in range(n_trees)]
#     #
#     # ("depths {}".format(depths))
#     #trees = [generate_random_tree(x, n_classes, depths[i], split_distribution, split_param, rng) for i in range(n_trees)]
#     trees = Parallel(n_jobs=50)(delayed(generate_random_tree)(x, n_classes, depths[i], split_distribution, split_param, rng) for i in range(n_trees))
#     forest = Forest(trees, rng)

#     return forest

# def generate_labels_random_forest(x, n_classes=2, n_trees=5, max_depth=5, depth_distribution="constant",
#                                        split_distribution="uniform", split_param=1, rng=None):
#     forest = generate_random_forest(x, n_classes, n_trees, max_depth, depth_distribution,
#                                     split_distribution, split_param, rng)

#     return forest.predict(x)


class GaussianNoise(nn.Module):
    def __init__(self, std, device):
        super().__init__()
        self.std = std
        self.device=device

    def forward(self, x):
        return x + torch.normal(torch.zeros_like(x), self.std)
    
    
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



def get_batch(batch_size, seq_len, num_features_max, num_features_sampler, hyperparameters, device=default_device, num_outputs=1, sampling='normal'
              , epoch=None, **kwargs):
    if 'multiclass_type' in hyperparameters and hyperparameters['multiclass_type'] == 'multi_node':
        num_outputs = num_outputs * hyperparameters['num_classes']

    correlation_strength = np.random.uniform(hyperparameters['correlation_strength_min'], hyperparameters['correlation_strength_max'])
    correlation_proba = np.random.uniform(hyperparameters['correlation_proba_min'], hyperparameters['correlation_proba_max'])
    def get_seq():
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
            # convert to tensor
            #data = torch.from_numpy(x).to(device)
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
        
        if np.random.random() < 0.0001:
            print("------------------")
            print("X")
            print(X.shape)
            print(X)
            print("data")
            print(data.shape)
            print(data)
            print("------------------")
        
        # Sample max depth from uniform distribution
        max_depth = 2 + int(np.random.exponential( 1./ hyperparameters['max_depth_lambda']))
        # Sample number of trees from an exponential distribution with parameter lambda
        n_estimators = (1 + int(np.random.exponential(1. / hyperparameters['n_estimators_lambda']))) if (hyperparameters['n_estimators'] is None) else hyperparameters['n_estimators']
        time_forest = time.time()
        forest = ExtraTreesClassifier(max_depth=max_depth, n_estimators=n_estimators, max_features=1, n_jobs=1) # max_features=1 means that the splits are totally random
        num_classes = hyperparameters['num_classes']        

        fake_y = np.random.randint(0, num_classes, data.shape[0])
        # label encoder to prevent holes in the class labels (e.g. 0, 1, 3)
        le = LabelEncoder()
        fake_y = le.fit_transform(fake_y)
        forest.fit(X,fake_y) #TODO unbalance
        
        if hyperparameters["randomize_leaves"]:
            forest = randomize_leaves_func(forest, num_classes)

        if hyperparameters["return_classes"]:
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
        
        # # Pad data with zeros to have 100 features
        # if data.shape[-1] < 100:
        #     padding = torch.zeros((data.shape[0], 100 - data.shape[-1]), device=device)
        #     data = torch.cat([data, padding], -1)

        return data.reshape(-1, 1, num_features).float(), torch.from_numpy(y).reshape(-1, 1, num_outputs).float()

    

    # if hyperparameters.get('new_forest_per_example', False):
    #     get_model = lambda: generate_random_forest(hyperparameters)
    # else:
    #     model = generate_random_forest(hyperparameters)
    #     get_model = lambda: model

    sample = [get_seq() for _ in range(0, batch_size)]

    x, y = zip(*sample)
    y = torch.cat(y, 1).detach().squeeze(2)
    #x = torch.cat(x, 1).detach()
    # concat x and pad with zeros
    # first pad with zeros to num_features_max
    x = [torch.cat([x[i], torch.zeros((seq_len, 1, num_features_max - x[i].shape[-1]))], -1) for i in range(len(x))]
    # then concat
    x = torch.cat(x, 1).detach()
    
    
    return x, y, y


DataLoader = get_batch_to_dataloader(get_batch)

