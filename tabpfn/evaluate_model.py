import numpy as np
import openml

import openml
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import os
import pickle
from tabpfn.scripts.transformer_prediction_interface import TabPFNClassifier
import pandas as pd
import torch
from create_model import load_model_no_train

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

def import_open_ml_data(dataset_id=None, task_id=None, remove_nans=None, impute_nans=None, categorical=False, regression=False, balance=False, rng=None) -> pd.DataFrame:
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



def get_validation_performance(model, metric="accuracy", suites=[337, 334],
                               baselines=[("gbt", GradientBoostingClassifier), 
                                          ("tabpfn", TabPFNClassifier),
                                          ("logreg", LogisticRegression)],
                               recompute=False):
    res = {}
    for suite_id in suites:
        suite = openml.study.get_suite(suite_id)
        tasks = suite.tasks
        # check if saved results exist
        if os.path.exists("results_{}.pkl".format(suite_id)) and not recompute:
            with open("results_{}.pkl".format(suite_id), "rb") as f:
                results_baselines = pickle.load(f)
        else:
            results_baselines = {}
        results_model = {}
        accepted_tasks = []
        for task_id in tasks:
            print("Task id: {}".format(task_id))
            X, y, _ = import_open_ml_data(task_id=task_id, remove_nans=True, impute_nans=False, categorical=False, regression=False, balance=False, rng=None)
            if X.shape[1] > 100:
                print("skipping task {} because it has too many features".format(task_id))
                continue
            if X is None:
                print("skipping task {} because it has no features".format(task_id))
                continue
            accepted_tasks.append(task_id)
            # Truncate the dataset to 10000 samples
            # Create a random state
            rng = np.random.RandomState(42)
            indices = rng.choice(X.shape[0], min(10000, X.shape[0]), replace=False)
            X = X[indices]
            y = y[indices]
            print("X0, Y0", X[0], y[0])
            # evaluate model
            # with cross validation
            # Create a cross-validation object
            #TODO change when not using my benchmark
            X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, train_size=1024, random_state=rng)
            # Evaluate the model by computing the accuracy
            model.fit(X_train, y_train)
            mean_score = model.score(X_test, y_test)
            print(f"Model Accuracy: {mean_score}")
            results_model[task_id] = mean_score
            # Compute the scores for baselines if not already computed
            if task_id not in results_baselines:
                results_baselines[task_id] = {}
                for name, baseline in baselines:
                    if name not in results_baselines[task_id]:
                        print("Computing baseline {}".format(name))
                        if name == "tabpfn":
                            baseline = baseline(device="cuda:0")
                        else:
                            baseline = baseline()
                        baseline.fit(X_train, y_train)
                        results_baselines[task_id][name] = baseline.score(X_test, y_test)
        # save results
        with open("results_{}.pkl".format(suite_id), "wb") as f:
            pickle.dump(results_baselines, f)
        
        # aggregate results
        # Make a dataframe with the results
        # with columns: task_id, model, score
        df = pd.DataFrame(columns=['task_id', 'model', 'score'])
        for task_id in accepted_tasks:
            df = df.append({'task_id': task_id, 'model': 'model', 'score': results_model[task_id]}, ignore_index=True)
            for name, baseline in baselines:
                df = df.append({'task_id': task_id, 'model': name, 'score': results_baselines[task_id][name]}, ignore_index=True)
        
        # Compute the mean rank of each model across all tasks
        # group by model and task_id, and compute the mean accuracy
        mean_acc = df.groupby(['model', 'task_id']).mean()


        # sort the resulting dataframe by model and accuracy
        sorted_acc = mean_acc.sort_values(['model', 'score'], ascending=[True, False])

        # assign ranks to each model based on sorted order of accuracy
        sorted_acc['rank'] = sorted_acc.groupby('task_id')['score'].rank(method='dense', ascending=False)


        # compute the mean rank for each model
        mean_ranks = sorted_acc.groupby('model')['rank'].mean()
        mean_rank = mean_ranks['model']

        # Compute the mean score of the model across all tasks
        mean_score = df[df['model'] == 'model']['score'].mean()
        
        # For each task, normalize the score of the model to be 0 for the worst score and 1 for the best score
        # Normalize the score for each model in each task
        df['normalized_score'] = (df['score'] - df.groupby(['task_id'])['score'].transform('min')) / (df.groupby(['task_id'])['score'].transform('max') - df.groupby(['task_id'])['score'].transform('min'))
        # Group by model, and compute the mean normalized score
        mean_score_normalized = df.groupby('model')['normalized_score'].mean()["model"]
        
        res_suite =  {f"mean_rank_{suite_id}": mean_rank, 
                         f"mean_score_{suite_id}": mean_score, 
                         f"mean_score_normalized_{suite_id}": mean_score_normalized}
        # Add accuracy for each task
        for task_id in accepted_tasks:
            res_suite[f"accuracy_{task_id}"] = results_model[task_id]
        res.update(res_suite)
        
    return res

if __name__ == """__main__""":
    get_validation_performance(model=TabPFNClassifier(device="cuda:0"))
    
        
        
        

        
        
        
        
            
            
            
            
            
        

