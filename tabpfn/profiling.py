from priors.trees import generate_random_forest
import numpy as np

x = np.random.normal(0, 1, (1152, 93))
sampling = "mixed"
seq_len = 1152
num_features = 93
device = 'cuda:1'
batch_size = 4
import torch
import random
def get_seq():
    if sampling == 'normal':
        data = torch.normal(0., 1., (seq_len, 1, num_features), device='cpu').float()
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
    elif sampling == 'uniform':
        data = torch.rand((seq_len, 1, num_features), device="cpu")
    else:
        raise ValueError(f'Sampling is set to invalid setting: {sampling}.')

    forest = generate_random_forest(data.numpy().reshape(-1, num_features), n_classes=2, n_trees=50, max_depth=15, depth_distribution="constant")
    y = forest.predict(data.numpy().reshape(-1, num_features))
    data = data[..., (torch.arange(data.shape[-1], device="cpu")+random.randrange(data.shape[-1])) % data.shape[-1]]

    return torch.tensor(data).to(device).reshape(-1, 1, num_features), torch.tensor(y).to(device).reshape(-1, 1, 1)

    

    # if hyperparameters.get('new_forest_per_example', False):
    #     get_model = lambda: generate_random_forest(hyperparameters)
    # else:
    #     model = generate_random_forest(hyperparameters)
    #     get_model = lambda: model

sample = [get_seq() for _ in range(0, batch_size)]

x, y = zip(*sample)
y = torch.cat(y, 1).detach().squeeze(2)
x = torch.cat(x, 1).detach()