# Jupyter notebook converted to Python script.

"""
<a href="https://colab.research.google.com/github/vs-152/FL-Contributions-Incentives-Project/blob/main/ISO_CIFAR10_OR_FINAL.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
"""

import torch
import torch.nn as nn
import numpy as np
import pulp
import copy
import time
from sklearn.model_selection import StratifiedShuffleSplit
import torchvision
# from torchvision.datasets import CIFAR10
# import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler
from itertools import chain, combinations
from tqdm import tqdm
from scipy.special import comb
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import glob, nibabel as nib, pandas as pd
# from monai.data import CacheDataset, DataLoader
from monai.data import Dataset as MonaiDataset, DataLoader
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, Orientationd, ScaleIntensityd,
    RandFlipd, RandSpatialCropd, Compose, SelectItemsd
)


from utils import *
from models import *


# -----------------------------------------------------------
# 0.  Paths & meta-data
# -----------------------------------------------------------

BRATS_DIR   = "/mnt/c/Datasets/MICCAI_FeTS2022_TrainingData"
VAL_DIR     = "/mnt/c/Datasets/MICCAI_FeTS2022_ValidationData"
CSV_PATH    = f"{BRATS_DIR}/partitioning_1.csv"     # pick 1, 2 … or sanity
MODALITIES  = ["flair", "t1", "t1ce", "t2"]
LABEL_KEY   = "seg"  # BraTS tumour mask filename ending

# -----------------------------------------------------------
# 1.  Read partition file → mapping   {client_id: [subjIDs]}
# -----------------------------------------------------------
part_df          = pd.read_csv(CSV_PATH)
partition_map    = (
    part_df.groupby("Partition_ID")["Subject_ID"]
           .apply(list)
           .to_dict()
)
NUM_CLIENTS = len(partition_map)

# -----------------------------------------------------------
# 2.  Build a list of dicts – one per subject
# -----------------------------------------------------------
def build_records(subject_ids):
    recs = []
    for sid in subject_ids:
        subj_dir = f"{BRATS_DIR}/{sid}"
        rec = {m: f"{subj_dir}/{sid}_{m}.nii.gz"
               for m in MODALITIES}
        rec["seg"] = f"{subj_dir}/{sid}_{LABEL_KEY}.nii.gz"
        recs.append(rec)
    return recs

def build_val_records(val_dir):
    subjects = sorted(glob.glob(f"{val_dir}/FeTS2022_*_flair.nii.gz"))
    recs = []
    for flair_path in subjects:
        sid = flair_path.split("/")[-1].split("_flair")[0]
        subj_dir = f"{val_dir}/{sid}"
        rec = {m: f"{subj_dir}/{sid}_{m}.nii.gz" for m in MODALITIES}
        recs.append(rec)
    return recs

# -----------------------------------------------------------
# 3.  MONAI transform pipelines  (fixed)
# -----------------------------------------------------------
IMG_KEYS   = [m for m in MODALITIES]
ALL_KEYS   = IMG_KEYS + [LABEL_KEY]

train_tf = Compose([
    LoadImaged(keys=ALL_KEYS),
    EnsureChannelFirstd(keys=ALL_KEYS),
    Orientationd(keys=ALL_KEYS, axcodes="RAS"),
    ScaleIntensityd(keys=ALL_KEYS, minv=-1.0, maxv=1.0), # scale to [-1,1]. Diffusion Models do better if centered on a 0 mean
    SelectItemsd(keys=ALL_KEYS),
])

val_tf = Compose([
    LoadImaged(keys=MODALITIES),
    EnsureChannelFirstd(keys=MODALITIES),
    Orientationd(keys=MODALITIES, axcodes="RAS"),
    ScaleIntensityd(keys=MODALITIES, minv=-1.0, maxv=1.0),
    SelectItemsd(keys=MODALITIES),
])

# -----------------------------------------------------------
# 4.  Build per-client datasets & dataloaders
# -----------------------------------------------------------
CUT_OFF    = 4  # How many sites to discard.
train_datasets = {}     # {client_id: monai CacheDataset}
for cid, subj_list in partition_map.items():
    if cid > CUT_OFF:
        print(f"capping at {cid} for now")
        break
    records = build_records(subj_list)
    train_datasets[cid] = MonaiDataset(data=records, transform=train_tf)#CacheDataset(data=records, transform=train_tf, cache_rate=0.0)

# Output:
#   Loading dataset:  10%|██████▌                                                          | 52/511 [00:42<06:16,  1.22it/s]

# -----------------------------------------------------------
# 5.  Build test dataset & dataloader
# -----------------------------------------------------------
val_records  = build_val_records(VAL_DIR)
test_dataset  = MonaiDataset(data=val_records, transform=val_tf) #, cache_rate=0.0)
# test_loader   = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

def test_inference(model, test_dataset):
    # --------- INFERENCE FOR SEGMENTATION
    model.eval()
    total_dice = 0.0
    testloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        for batch in testloader:
            images = torch.stack([batch[k] for k in ["flair", "t1", "t1ce", "t2"]], dim=1).squeeze(2).to(device)  # (B, 4, D, H, W)
            labels = batch["seg"].long().to(device)  # (B, 1, D, H, W)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1, keepdim=True)  # (B, 1, D, H, W)

            # Simple Dice calculation (per class optional)
            intersection = (preds == labels).sum().item()
            total = labels.numel()
            total_dice += 2.0 * intersection / (total + preds.numel())

    return total_dice / len(testloader)

# N = 10 #srch

# after you build train_datasets
idxs_users = np.array(sorted(train_datasets.keys()))
N = len(idxs_users)

# N = list(train_datasets.keys())[-1]
print(f"We got {N} clients")
local_bs = 512
lr = 0.01
local_ep = 5
EPOCHS = 5

# noise_rates = np.linspace(0, 1, N, endpoint=False)
# split_dset = mnist_iid(trainset, N)
# user_groups = {i: 0 for i in range(1, N+1)}
# noise_idx = {i: 0 for i in range(1, N+1)}
# train_datasets = {i: 0 for i in range(1, N+1)}
# for n in range(N):
#     user_groups[n+1] = np.array(list(split_dset[n]), dtype=np.int)
#     user_train_x, user_train_y = x_train[user_groups[n+1]], y_train[user_groups[n+1]]
#     user_noisy_y, noise_idx[n+1] = noisify_MNIST(noise_rates[n], 'symmetric', user_train_x, user_train_y)
    
#     train_datasets[n+1] = CustomTensorDataset((user_train_x, user_noisy_y), transform_train)

def copy_batchnorm_stats(subset_weights, global_model_state_dict):
    for pair_1, pair_2 in zip(subset_weights.items(), global_model_state_dict.items()):
        if ('running' in pair_1[0]) or ('batches' in pair_1[0]):
            subset_weights[pair_1[0]] = global_model_state_dict[pair_1[0]]
    
    return subset_weights


global_model = ResUNet3D(in_channels=4, out_channels=3).to(device) # Segmentation model used in FeTS

global_model.to(device)
global_model.train()

global_weights = global_model.state_dict()
powerset = list(powersettool(range(1, N+1)))

submodel_dict = {}  
submodel_dict[()] = copy.deepcopy(global_model)
accuracy_dict = {}
shapley_dict = {}

start_time = time.time()

for subset in range(1, N+1):
    submodel_dict[(subset,)] = copy.deepcopy(global_model)
    submodel_dict[(subset,)].to(device)
    submodel_dict[(subset,)].train() 
 
train_loss, train_accuracy = [], []
val_acc_list, net_list = [], []
print_every = 1

idxs_users = np.arange(1, N+1)
# total_data = sum(len(user_groups[i]) for i in range(1, N+1))
# fraction = [len(user_groups[i])/total_data for i in range(1, N+1)]

# ── collect dataset sizes ──────────────────────────────────────────────────
# MONAI's CacheDataset inherits __len__, so `len(ds)` is cheap:
sizes = {k: len(ds) for k, ds in train_datasets.items()}


# ── total samples across all clients ───────────────────────────────────────
total_data = sum(sizes.values())

# ── FedAvg weight (a.k.a. fraction) for each client ────────────────────────
# Keep the list in key order 1…N so it lines up with your loops later.
fraction = [sizes[i] / total_data for i in range(1, N + 1)]

# ───────────────────────────────────────────────────────────────────────────

for epoch in tqdm(range(EPOCHS)):
    local_weights, local_losses = [], []
    print(f'\n | Global Training Round : {epoch+1} |\n')
    global_model.train()
    for idx in idxs_users:
        trainloader = DataLoader(
            train_datasets[idx],
            batch_size=local_bs,          # or 1–2 for full volumes
            shuffle=True,
            num_workers=0,         
            pin_memory=False
        )

        
        local_model = LocalUpdateMONAI(
            lr=lr,
            local_ep=local_ep,
            trainloader=trainloader,
            img_keys=("flair","t1","t1ce","t2"),
            label_key="seg",
        )
        
        w, loss = local_model.update_weights(model=copy.deepcopy(global_model))
        local_weights.append(copy.deepcopy(w))
        local_losses.append(copy.deepcopy(loss))
    global_weights = average_weights(local_weights, fraction) 
    loss_avg = sum(local_losses) / len(local_losses)
    train_loss.append(loss_avg)

    print(f"[Round {epoch+1:02d}] global loss={loss_avg:.4f}")

    gradients = calculate_gradients(local_weights, global_model.state_dict()) 
    for i in range(1, N+1):
        subset_weights = update_weights_from_gradients(gradients[i-1], submodel_dict[(i,)].state_dict()) 
        subset_weights = copy_batchnorm_stats(subset_weights, global_model.state_dict())
        submodel_dict[(i,)].load_state_dict(subset_weights)

    global_model.load_state_dict(global_weights)
    global_model.eval()

    if (epoch+1) % print_every == 0:
        print(f' \nAvg Training Stats after {epoch+1} global rounds:')
        print(f'Training Loss : {np.mean(np.array(train_loss))}')
        # print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))

test_acc, test_loss = test_inference(global_model, test_dataset)
print(f' \n Results after {EPOCHS} global rounds of training:')
print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

accuracy_dict[powerset[-1]] = test_acc

# ADJUSTED-OR APPROX
for subset in powerset[:-1]: 
    if len(subset) > 1:
        # calculate the average of the subset of weights from list of all the weights
        subset_weights = average_weights([submodel_dict[(i,)].state_dict() for i in subset], [fraction[i-1] for i in subset]) 
        submodel = copy.deepcopy(submodel_dict[()])
        submodel.load_state_dict(subset_weights)
        
        test_acc, test_loss = test_inference(submodel,test_dataset)
        print(f' \n Results after {EPOCHS} global rounds of training (for OR): ')
        print("|---- Test Accuracy for {}: {:.2f}%".format(subset, 100*test_acc))
        accuracy_dict[subset] = test_acc
    else: 
        test_acc, test_loss = test_inference(submodel_dict[subset], test_dataset)
        accuracy_dict[subset] = test_acc

trainTime = time.time() - start_time
start_time = time.time()
shapley_dict = shapley(accuracy_dict, N)
shapTime = time.time() - start_time
start_time = time.time()
lc_dict = least_core(accuracy_dict, N)
LCTime = time.time() - start_time
totalShapTime = trainTime + shapTime
totalLCTime = trainTime + LCTime
print(f'\n ACCURACY: {accuracy_dict[powerset[-1]]}')
print('\n Total Time Shapley: {0:0.4f}'.format(totalShapTime))
print('\n Total Time LC: {0:0.4f}'.format(totalLCTime))

def stats(vector):
    n = len(vector)
    egal = np.array([1/n for i in range(n)])
    normalised = np.array(vector / vector.sum())
    msg = f'Original vector: {vector}\n'
    msg += f'Normalised vector: {normalised}\n'
    msg += f'Max Dif: {normalised.max()-normalised.min()}\n'
    msg += f'Distance: {np.linalg.norm(normalised-egal)}\n'

    msg += f'Budget: {vector.sum()}\n'
    print(msg)

print(stats(np.array(list(shapley_dict.values()))))

print(stats(np.array([i.value() for i in lc_dict.variables()])[1:]))