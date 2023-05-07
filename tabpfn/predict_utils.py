from sklearn.preprocessing import PowerTransformer
from tabpfn.scripts.transformer_prediction_interface import TabPFNClassifier
from create_model import get_model_no_train
import numpy as np
import torch

def predict(model_pytorch, X_train, X_test, y_train, preprocess=True, device="cpu"):
    num_features = model_pytorch.encoder.weight.shape[1]
    if X_train.shape[1] > num_features:
        return None
    if preprocess:
        X_train = PowerTransformer().fit_transform(X_train)
        X_test = PowerTransformer().fit_transform(X_test)
    y_test = np.zeros((X_test.shape[0], 1))
    X, y = torch.tensor(np.concatenate([X_train, X_test]).astype(np.float32)),\
        torch.tensor(np.concatenate([y_train, y_test]))
    # pad X to num_features
    X = torch.cat([X, torch.zeros((X.shape[0], num_features - X.shape[1]))], axis=1)
    # add a dimension 1 in the middle
    X = X.reshape(X.shape[0], 1, X.shape[1]).float()
    y = y.reshape(-1, 1).float()
    # replace the y after the single_eval_pos with 0
    single_eval_pos = len(X_train)
    # hide the test data to make sure it is not used
    y = torch.cat([y[:single_eval_pos], torch.zeros_like(y[single_eval_pos:])], axis=0)
    # move to device
    X, y = X.to(device), y.to(device)
    logits = model_pytorch((1, X, y), single_eval_pos=len(X_train))
    preds = logits.argmax(axis=2).detach().cpu().numpy().reshape(-1)
    return preds

def load_model(checkpoint, n_heads, device="cpu"):
    tabpfn = TabPFNClassifier()
    config=tabpfn.c
    del tabpfn
    config["use_wandb"] = False
    config["use_neptune"] = False
    config["wandb_offline"] = False
    config["curriculum"] = False
    # config["bptt"] = 1152
    # config["prior_type"] = "linear"
    # #useless
    config["curriculum_step"] = 10
    config["curriculum_tol"] = 0
    config["curriculum_start"] = 10
    config["get_openml_from_pickle"] = False
    config["validate_on_datasets"] = False
    config["num_workers"] = 2
    config["num_classes"] = 10
    config["max_num_classes"] = 10
    config["n_out"] = 10
    config["sampling"] = "normal"
    config["remove_outliers_in_flexible_categorical"] = False
    config["normalize_x_in_flexible_categorical"] = False
    config["random_feature_rotation"] = False
    config["nhead"] = n_heads
    config["random_feature_rotation"] = False
    checkpoint_file = f"model_{checkpoint}"
    path = f"tabpfn/model_checkpoints/{checkpoint_file}.pt"
    print(f'Loading checkpoint file {checkpoint_file}')
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
    else:
        model_state = loaded_data
        
    # remove the "module." prefix from keys
    model_state = {k.replace("module.", ""): v for k, v in model_state.items()}
    config["num_features"] = model_state["encoder.weight"].shape[1] 
    print("num_features", config["num_features"]) 
    #config["num_features_used"] = {'num_features_func': uniform_int_sampler_f(3, config["num_features_no_pad"])} #TODO get rid of differentiable
    from tabpfn.utils import get_no_op_scheduler
    scheduler = get_no_op_scheduler

    from scripts.model_builder import get_model    
    transformer = get_model(config, device, should_train=False, state_dict=model_state, 
                            scheduler=get_no_op_scheduler)[2]
    transformer = transformer.to(device)
    transformer.eval()
    return transformer

class TabPFNWrapper(object):
    def __init__(self, checkpoint, n_heads, preprocess=True, device="cpu"):
        self.model = load_model(checkpoint, n_heads, device)
        self.X_train = None
        self.y_train = None
        self.is_fitted = False
        self.device = device
    def fit(self, X_train, y_train):
        # store the data
        self.X_train = X_train
        self.y_train = y_train
        self.is_fitted = True
    def predict(self, X):
        assert self.is_fitted
        return predict(self.model, self.X_train, X, self.y_train, None, preprocess=True, 
                       device=self.device)
