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
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer



def get_batch(batch_size, seq_len, num_features, hyperparameters, device=default_device, num_outputs=1, sampling='normal'
              , epoch=None, **kwargs):
    if 'multiclass_type' in hyperparameters and hyperparameters['multiclass_type'] == 'multi_node':
        num_outputs = num_outputs * hyperparameters['num_classes']

    correlation_strength = np.random.uniform(hyperparameters['correlation_strength_min'], hyperparameters['correlation_strength_max'])
    correlation_proba = np.random.uniform(hyperparameters['correlation_proba_min'], hyperparameters['correlation_proba_max'])

    def get_seq():
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
            number_of_features_to_remove = int(np.random.uniform(0, hyperparameters["random_feature_removal"] * X.shape[1]))
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
        forest.fit(X, np.random.randint(0, num_classes, data.shape[0])) #TODO unbalance
            
        y = forest.predict_proba(X)[:, 0]
        time_forest = time.time() - time_forest
        if np.random.random() < 0.001:
            print("Time to fit forest: ", time_forest)

        data = torch.from_numpy(data)#.to(device)
        # random feature rotation
        if hyperparameters.get("random_feature_rotation", False):
            data = data[..., (torch.arange(data.shape[-1], device="cpu")+random.randrange(data.shape[-1])) % data.shape[-1]]
        

        return data.reshape(-1, 1, num_features).float(), torch.from_numpy(y).reshape(-1, 1, num_outputs).float()

    

    sample = [get_seq() for _ in range(0, batch_size)]

    x, y = zip(*sample)
    y = torch.cat(y, 1).detach().squeeze(2)
    x = torch.cat(x, 1).detach()
    
    return x, y, y


DataLoader = get_batch_to_dataloader(get_batch)

