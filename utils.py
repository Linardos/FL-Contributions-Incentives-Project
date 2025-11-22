import torch
import torch.nn as nn
import numpy as np
import pulp
import copy
import time
from sklearn.model_selection import StratifiedShuffleSplit
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler
from itertools import chain, combinations
from tqdm import tqdm
from scipy.special import comb
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def print_solution(model):
    """Prints solution of the model nicely!"""

    print(f"status: {model.status}, {pulp.LpStatus[model.status]}")
    print(f"objective: {model.objective.value()}")
    for var in model.variables():
        print(f"{var.name}: {round(var.value(),3)}")

def noisify_MNIST(noise_rate, noise_type, x, y, perm=[], **kwargs):
    '''Returns a symmetrically noisy dataset
    or a an asymmetrically noisy dataset with permutation matrix perm.
    '''
    if (noise_rate == 0.):
        return y, []
    if 'seed' in kwargs:
        _, noise_idx = next(
            iter(StratifiedShuffleSplit(
                n_splits=1,
                test_size=noise_rate,
                random_state=kwargs['seed']).split(x, y)))
    else:
        _, noise_idx = next(iter(StratifiedShuffleSplit(
            n_splits=1, test_size=noise_rate).split(x, y)))
    y_noisy = y.copy()
    if (noise_type == 'symmetric'):
        for i in noise_idx:
            t1 = np.arange(10)
            t2 = np.delete(t1, y[i])
            y_noisy[i] = np.random.choice(t2, 1)
    elif (noise_type == 'asymmetric'):
        pure_noise = perm[y]
        for i in noise_idx:
            if (perm[y[i]] == y[i]):
                noise_idx = np.delete(noise_idx, np.where(noise_idx == i))
            else:
                y_noisy[i] = pure_noise[i]

    return y_noisy, noise_idx

def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])

    return dict_users

def average_weights(w, fraction):  # this can also be used to average gradients
    """
    :param w: list of weights generated from the users
    :param fraction: list of fraction of data from the users
    :Returns the weighted average of the weights.
    """
    w_avg = copy.deepcopy(w[0]) #copy the weights from the first user in the list 
    for key in w_avg.keys():
        w_avg[key] *= torch.tensor(fraction[0]/sum(fraction), dtype=w_avg[key].dtype)
        for i in range(1, len(w)):
            w_avg[key] += w[i][key] * torch.tensor(fraction[0]/sum(fraction), dtype=w_avg[key].dtype)

    return w_avg

def calculate_gradients(new_weights, old_weights):
    """
    :param new_weights: list of weights generated from the users
    :param old_weights: old weights of a model, probably before training
    :Returns the list of gradients.
    """
    gradients = []
    for i in range(len(new_weights)):
        gradients.append(copy.deepcopy(new_weights[i]))
        for key in gradients[i].keys():
            gradients[i][key] -= old_weights[key]

    return gradients

def update_weights_from_gradients(gradients, old_weights):
    """
    :param gradients: gradients
    :param old_weights: old weights of a model, probably before training
    :Returns the updated weights calculated by: old_weights+gradients.
    """
    updated_weights = copy.deepcopy(old_weights)
    for key in updated_weights.keys():
        updated_weights[key] = old_weights[key] + gradients[key]

    return updated_weights


def powersettool(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def least_core(char_function_dict, N):
    """Solves the least core LP problem.

    Args:
        N: number of participants.
        char_function_dict: dictionary with participants as keys and 
        corresponding characteristic function value as values
    """
    model = pulp.LpProblem('least_core', pulp.LpMinimize)
    x = {i: pulp.LpVariable(name=f'x({i})', lowBound=0) for i in range(1, N+1)}
    e = pulp.LpVariable(name='e')
    model += e # decision variable
    grand_coalition = tuple(i for i in range(1, N+1))
    model += pulp.lpSum(x) == char_function_dict[grand_coalition]
    for key, value in char_function_dict.items():
        model += pulp.lpSum(x[idx] for idx in key) + e >= value
    model.solve()
    print_solution(model)

    return model

def shapley(utility, N):

    shapley_dict = {}
    for i in range(1, N+1):
        shapley_dict[i] = 0
    for key in utility:
        if key != ():
            for contributor in key:
                print('contributor:', contributor, key) # print check
                marginal_contribution = utility[key] - utility[tuple(i for i in key if i!=contributor)]
                print('marginal:', marginal_contribution) # print check
                shapley_dict[contributor] += marginal_contribution /((comb(N-1,len(key)-1))*N)

    return shapley_dict