import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.inspection import DecisionBoundaryDisplay
from tabpfn.scripts.transformer_prediction_interface import TabPFNClassifier

import random
import time
import warnings
from datetime import datetime

import torch

import numpy as np

import matplotlib.pyplot as plt
from tabpfn.scripts.differentiable_pfn_evaluation import eval_model_range
from tabpfn.scripts.model_builder import get_model, get_default_spec, save_model, load_model
from tabpfn.scripts.transformer_prediction_interface import transformer_predict, get_params_from_config, load_model_workflow

from tabpfn.scripts.model_configs import *

from tabpfn.datasets import load_openml_list, open_cc_dids, open_cc_valid_dids
from tabpfn.priors.utils import plot_prior, plot_features
from tabpfn.priors.utils import uniform_int_sampler_f

from tabpfn.scripts.tabular_metrics import calculate_score_per_method, calculate_score
from tabpfn.scripts.tabular_evaluation import evaluate

from tabpfn.priors.differentiable_prior import DifferentiableHyperparameterList, draw_random_style, merge_style_with_info
from tabpfn.scripts import tabular_metrics
from tabpfn.notebook_utils import *
import argparse
import wandb 
import torch

def load_model(path, filename, device, config_sample, verbose=0):
    # TODO: This function only restores evaluation functionality but training canät be continued. It is also not flexible.
    # print('Loading....')
    model_state = torch.load(
        os.path.join(path, filename), map_location='cpu')
    if ('differentiable_hyperparameters' in config_sample
            and 'prior_mlp_activations' in config_sample['differentiable_hyperparameters']):
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

    #print('Memory', str(get_gpu_memory()))

    model = get_model(config_sample, device=device, should_train=False, verbose=verbose)
    module_prefix = 'module.'
    model_state = {k.replace(module_prefix, ''): v for k, v in model_state.items()}
    model[2].load_state_dict(model_state)
    model[2].to(device)
    model[2].eval()

    return model, config_sample

large_datasets = True
max_samples = 10000 if large_datasets else 5000
bptt = 10000 if large_datasets else 3000
suite='cc'
base_path = '.'
max_features = 100

args = argparse.Namespace()
args.prior = "trees"
args.task_type = 'multiclass'
args.device = 4
args.wandb = False
args.name = "test"
args.save_every = 59

device = 'cuda:{}'.format(args.device) if args.device >= 0 else 'cpu'


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
    config['balanced'] = False
    model_string = model_string + '_multiclass'
    
    model_string = model_string + '_'+datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    
    return config, model_string

config, model_string = reload_config(longer=1)

config['bptt_extra_samples'] = None

config["sampling"] = "mixed"
del config['differentiable_hyperparameters']['sampling']
config["num_classes"] = 2

if config['prior_type'] == 'trees':
    config["n_trees"] = 25
    config["max_depth"] = 10
    config["depth_distribution"] = "uniform"
    config["split_distribution"] = "uniform"
    config["split_param"] = 1
elif config['prior_type'] == 'mlp':
    config['output_multiclass_ordered_p'] = 0.
    del config['differentiable_hyperparameters']['output_multiclass_ordered_p']

    config['multiclass_type'] = 'rank'
    del config['differentiable_hyperparameters']['multiclass_type']

    config['pre_sample_causes'] = True



config['multiclass_loss_type'] = 'nono' # 'compatible'
config['normalize_to_ranking'] = False # False

config['categorical_feature_p'] = .2 # diff: .0

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
config['batch_size'] = 8*config['aggregate_k_gradients']
config['num_steps'] = 1024//config['aggregate_k_gradients']
config['epochs'] = 400
config['total_available_time_in_s'] = None #60*60*22 # 22 hours for some safety...

config['train_mixed_precision'] = True
config['efficient_eval_masking'] = True

config["use_wandb"] = args.wandb
config["name"] = args.name
config["save_every"] = args.save_every

if args.wandb == True:
    wandb.init(project="tabpfn_training", entity="leogrin")
    wandb.config.update(config)

config_sample = evaluate_hypers(config)



names = [
    "Nearest Neighbors",
    "Linear SVM",
    "RBF SVM",
    "Gaussian Process",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    "Naive Bayes",
    "QDA",
    "TabPFN",
    "TabPFN_trees",
    "TabPFN_trees_prepro"
]


tabpfn_trees = TabPFNClassifier(no_preprocess_mode=True)
checkpoint = "trees456_390"
model_pytorch = load_model("tabpfn/model_checkpoints", f"model_{checkpoint}.pt", 2, config_sample, 0)[0]
tabpfn_trees.model = model_pytorch

# tabpfn_trees_prepro = TabPFNClassifier(no_preprocess_mode=False)
# checkpoint = "trees_456_390"
# model_pytorch_propro = load_model("tabpfn/model_checkpoints", f"model_{checkpoint}.pt", 2, config_sample, 0)[0]
# tabpfn_trees_prepro.model = model_pytorch_propro

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
    TabPFNClassifier(),
    tabpfn_trees,
    #tabpfn_trees_prepro
]

def make_square(inner_size=0.5, outer_size=1.0, n_samples=150, random_state=None):
    rng = np.random.default_rng(seed=random_state)

    n_inner = n_samples // 2
    n_outer = n_samples - n_inner

    inner_points = rng.uniform(low=-inner_size/2, high=inner_size/2, size=(n_inner, 2))
    outer_theta = rng.uniform(low=0, high=2 * np.pi, size=n_outer)
    outer_radii = rng.uniform(low=inner_size / 2, high=outer_size / 2, size=n_outer)
    outer_points = np.vstack([outer_radii * np.cos(outer_theta), outer_radii * np.sin(outer_theta)]).T

    X = np.vstack([inner_points, outer_points])
    y = np.hstack([np.zeros(n_inner), np.ones(n_outer)])

    return X, y

def make_L_shape(n_samples=150, boundary_ratio=0.5, random_state=None):
    rng = np.random.default_rng(seed=random_state)

    X = rng.uniform(low=0, high=1, size=(n_samples, 2))
    
    y = np.zeros(n_samples, dtype=int)
    y[(X[:, 0] > boundary_ratio) & (X[:, 1] > boundary_ratio)] = 1

    return X, y


import numpy as np
import matplotlib.pyplot as plt

def generate_stripes(samples, stripe_width):
    # generate X uniformly in 2D
    X = np.random.uniform(0, 1, (samples, 2))
    # generate y by checking if x is in the stripe
    y = np.zeros(samples)
    for i in range(samples):
        if X[i, 0] // stripe_width % 2 == 0:
            y[i] = 1
    return X, y
        



def make_diagonal(n_samples=150, margin=0.1, random_state=None):
    rng = np.random.default_rng(seed=random_state)

    X = rng.uniform(low=0, high=1, size=(n_samples, 2))
    y = (X[:, 1] > X[:, 0] + margin / 2).astype(int)

    return X, y





X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1
)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)

datasets = [
    make_moons(noise=0.3, random_state=0),
    make_circles(noise=0.2, factor=0.5, random_state=1),
    linearly_separable,
    make_square(random_state=0),
    make_L_shape(random_state=0),
    make_diagonal(random_state=0),
    generate_stripes(250, 0.1)
]

figure = plt.figure(figsize=(27, 9))
i = 1
# iterate over datasets
for ds_cnt, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    X, y = ds
    print("Shapes: ", X.shape, y.shape)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42
    )

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(["#FF0000", "#0000FF"])
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    if ds_cnt == 0:
        ax.set_title("Input data")
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k")
    # Plot the testing points
    ax.scatter(
        X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors="k"
    )
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        print("Classifier: ", name)
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)

        clf = make_pipeline(StandardScaler(), clf)
        print("X_train: ", X_train.shape)
        print("X_test: ", X_test.shape)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        DecisionBoundaryDisplay.from_estimator(
            clf, X, cmap=cm, alpha=0.8, ax=ax, eps=0.5
        )

        # Plot the training points
        ax.scatter(
            X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k"
        )
        # Plot the testing points
        ax.scatter(
            X_test[:, 0],
            X_test[:, 1],
            c=y_test,
            cmap=cm_bright,
            edgecolors="k",
            alpha=0.6,
        )

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(())
        ax.set_yticks(())
        if ds_cnt == 0:
            ax.set_title(name)
        ax.text(
            x_max - 0.3,
            y_min + 0.3,
            ("%.2f" % score).lstrip("0"),
            size=15,
            horizontalalignment="right",
        )
        i += 1

plt.tight_layout()
plt.show()
