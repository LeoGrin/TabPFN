import numpy as np
import openml

import openml
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import os
import pickle
from tabpfn.scripts.transformer_prediction_interface import TabPFNClassifier
import pandas as pd
import torch
from create_model import load_model_no_train
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from scipy.stats import special_ortho_group


def balance_data(x, y):
    rng = np.random.RandomState(0)
    print("Balancing")
    print(x.shape)
    indices = [(y == i) for i in np.unique(y)]
    sorted_classes = np.argsort(
        list(map(sum, indices)))  # in case there are more than 2 classes, we take the two most numerous

    n_samples_min_class = sum(indices[sorted_classes[-2]])
    print("n_samples_min_class", n_samples_min_class)
    indices_max_class = rng.choice(np.where(indices[sorted_classes[-1]])[0], n_samples_min_class, replace=False)
    indices_min_class = np.where(indices[sorted_classes[-2]])[0]
    total_indices = np.concatenate((indices_max_class, indices_min_class))
    y = y[total_indices]
    indices_first_class = (y == sorted_classes[-1])
    indices_second_class = (y == sorted_classes[-2])
    y[indices_first_class] = 0
    y[indices_second_class] = 1

    return x.iloc[total_indices], y

def import_open_ml_data(dataset_id=None, task_id=None, remove_nans=None, impute_nans=None, categorical=False, regression=False, balance=False, rng=None,
                        one_hot_encoding=False) -> pd.DataFrame:
    """
    Import data from openML
    :param int openml_task_id:
    :param path_to_file:
    :return:
    """
    if task_id is not None:
        task = openml.tasks.get_task(task_id)  # download the OpenML task
        dataset = task.get_dataset()
    elif dataset_id is not None:
        dataset = openml.datasets.get_dataset(dataset_id)
    # retrieve categorical data for encoding
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="dataframe", target=dataset.default_target_attribute
    )
    categorical_indicator = np.array(categorical_indicator)
    print("{} categorical columns".format(sum(categorical_indicator)))
    print("{} columns".format(X.shape[1]))
    y_encoder = LabelEncoder()

    # Replace categorical values by integers for each categorical column
    
    for i, categorical in enumerate(categorical_indicator):
        #TODO check 
       X[X.columns[i]] =X[X.columns[i]].astype('category')
       X[X.columns[i]] =X[X.columns[i]].cat.codes
       X[X.columns[i]] =X[X.columns[i]].astype('int64')

    # remove missing values
    assert remove_nans or impute_nans, "You need to remove or impute nans"
    if remove_nans:
        missing_rows_mask = X.isnull().any(axis=1)
        if sum(missing_rows_mask) > X.shape[0] / 5:
            print("Removed {} rows with missing values on {} rows".format(
                sum(missing_rows_mask), X.shape[0]))
        X = X[~missing_rows_mask]
        y = y[~missing_rows_mask]
        n_rows_non_missing = X.shape[0]
        if n_rows_non_missing == 0:
            print("Removed all rows")
            return None
    elif impute_nans:
        from sklearn.impute import SimpleImputer
        # Impute numerical columns with mean and categorical columns with most frequent
        categorical_imputer = SimpleImputer(strategy="most_frequent")
        numerical_imputer = SimpleImputer(strategy="mean")
        # check that there a > 0 categorical columns
        if sum(categorical_indicator) > 0:
            X[X.columns[categorical_indicator]] = categorical_imputer.fit_transform(X.iloc[:, categorical_indicator])
        # check that there a > 0 numerical columns
        if sum(~categorical_indicator) > 0:
            X[X.columns[~categorical_indicator]] = numerical_imputer.fit_transform(X.iloc[:, ~categorical_indicator])

    if one_hot_encoding:
        transformer = ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(), [col for col in X.columns if X[col].dtype == 'category']),
            ('passthrough', 'passthrough', [col for col in X.columns if X[col].dtype != 'category']),
        ])

        # create a pipeline with the column transformer
        pipeline = Pipeline([
            ('transformer', transformer)
        ])

        # fit and transform the data
        X_transformed = pipeline.fit_transform(X)
            

    # print("removing {} categorical features among {} features".format(sum(categorical_indicator), X.shape[1]))
    # X = X.to_numpy()[:, ~categorical_indicator]  # remove all categorical columns
    # if X.shape[1] == 0:
    #     print("removed all features, skipping this task")
    #     return None

    y = y_encoder.fit_transform(y)



    if regression:
        y = y.astype(np.float64)
    else:
        y = y.astype(np.int64)

    if balance:
        X, y = balance_data(X, y)

    X = X.to_numpy()

    if categorical:
        return X, y, categorical_indicator

    return X, y, None



def get_benchmark_performance(model, metric="accuracy", suites=[337, 334, "cc18"], 
                              tasks_per_suite = [None, None, [11,
                                                                14,
                                                                15,
                                                                16,
                                                                18,
                                                                22,
                                                                23,
                                                                29,
                                                                31,
                                                                37,
                                                                50,
                                                                54,
                                                                188,
                                                                458,
                                                                469,
                                                                1049,
                                                                1050,
                                                                1063,
                                                                1068,
                                                                1510,
                                                                1494,
                                                                1480,
                                                                1462,
                                                                1464,
                                                                6332,
                                                                23381,
                                                                40966,
                                                                40982,
                                                                40994,
                                                                40975]],
                               recompute=False, n_iter=3,
                               model_name = "None",
                               one_hot_encoding=False,
                               random_rotation=False,):
    res = pd.DataFrame(columns=["suite_id", "task_id", "seed", "metric", "value"])
    for i, suite_id in enumerate(suites):
        if tasks_per_suite[i] is None:
            suite = openml.study.get_suite(suite_id)
            tasks = suite.tasks
        else:
            tasks = tasks_per_suite[i]
        # check if saved results exist
        if os.path.exists("benchmark_results_{}.pkl".format(suite_id)) and not recompute:
            with open("benchmark_results_{}.pkl".format(suite_id), "rb") as f:
                results_baselines = pickle.load(f)
        else:
            results_baselines = {}
        accepted_tasks = []
        #suite_id = str(suite_id) + "_large"
        for task_id in tasks:
            for seed in range(n_iter):
                print("Task id: {}".format(task_id))
                try:
                    if suite_id == "cc18":
                        X, y, categorical_indicator = import_open_ml_data(dataset_id=task_id, remove_nans=True, impute_nans=False, categorical=True, regression=False, balance=False, rng=None,
                                                                          one_hot_encoding=one_hot_encoding)
                    else:
                        X, y, categorical_indicator = import_open_ml_data(task_id=task_id, remove_nans=True, impute_nans=False, categorical=True, regression=False, balance=False, rng=None,
                                                                          one_hot_encoding=one_hot_encoding)
                except Exception as e:
                    print(f"Error while loading task {task_id}: {e}")
                    continue
                if X.shape[1] > 100:
                    print("skipping task {} because it has too many features".format(task_id))
                    continue
                if X is None:
                    print("skipping task {} because it has no features".format(task_id))
                    continue
                if model_name == "hgbt":
                    print("using hgbt")
                    print("categorical", categorical_indicator)
                    model = HistGradientBoostingClassifier(categorical_features=categorical_indicator)
                accepted_tasks.append(task_id)
                # Truncate the dataset to 10000 samples
                rng = np.random.RandomState(seed)
                indices = rng.choice(X.shape[0], min(10000, X.shape[0]), 
                                        replace=False)
                X = X[indices]
                y = y[indices]
                print(X[0])
                print(y[0])
                
                if random_rotation:
                    rotation_matrix = special_ortho_group.rvs(X.shape[1], random_state=rng)
                    X = X @ rotation_matrix
                
                # evaluate model
                # with cross validation
                # Create a cross-validation object
                #TODO change when not using my benchmark
                # X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=rng)
                # if X_train.shape[0] > 10000:
                #     indices = rng.choice(X_train.shape[0], 10000, replace=False)
                #     X_train = X_train[indices]
                #     y_train = y_train[indices]
                # if X_test.shape[0] > 10000:
                #     indices = rng.choice(X_test.shape[0], 10000, replace=False)
                #     X_test = X_test[indices]
                #     y_test = y_test[indices]
                # print("X_train", X_train.shape)
                if len(X) > 1300:
                    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, train_size=1024, random_state=rng)
                else:
                    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=rng)
                #Evaluate the model by computing the accuracy
                model.fit(X_train, y_train)#, overwrite_warning=True)
                y_pred = model.predict(X_test)
                score = accuracy_score(y_test, y_pred)
                print("accuracy", score)
                res = pd.concat([res, pd.DataFrame({"suite_id": [suite_id], "task_id": [task_id], "seed": [seed], "metric": ["accuracy"], "value": [score]})])
                y_pred_proba = model.predict_proba(X_test)
                if y_pred_proba.shape[1] == 2:
                    y_pred_proba = y_pred_proba[:, 1]
                    multi_class = "raise"
                else:
                    multi_class = "ovo"
                score = roc_auc_score(y_test, y_pred_proba, multi_class=multi_class)
                res = pd.concat([res, pd.DataFrame({"suite_id": [suite_id], "task_id": [task_id], "seed": [seed], "metric": ["roc_auc"], "value": [score]})])
                    
        
        
    return res

if __name__ == """__main__""":
    device = "cuda:3"
    #checkpoint = "trees55166_49voozm8_220"
    #checkpoint = "trees69859_eouc70o7_390"
    #checkpoint = "trees676_obqe7mfl_350"
    #checkpoint = "trees4315_080m7u0l_390"
    checkpoint = "trees97149_stz4qj1z_180"
    model = TabPFNClassifier(device=device, no_preprocess_mode=True)
    #model = TabPFNClassifier(device=device)
    #model = GradientBoostingClassifier()
    #model = MLPClassifier()
    model_pytorch = load_model_no_train("model_checkpoints", f"model_{checkpoint}.pt", 0, model.c, 0)[0]
    model.model = model_pytorch
    res = get_benchmark_performance(model, model_name="tabpfn", one_hot_encoding=False, 
                                    random_rotation=False)
    #model_name = f"tabpfn_{checkpoint}"
    #model_name = "mlp_sklearn"
    model_name = checkpoint
    #model_name = "gbt"
    print(res)
    res["model"] = model_name
    results = pd.read_csv("results_benchmark.csv")
    # remove old results
    results = results[results["model"] != model_name]
    # add new results
    results = pd.concat([results, res])
    results.to_csv(f"results_benchmark.csv")
    
        
        
        

        
        
        
        
            
            
            
            
            
        

