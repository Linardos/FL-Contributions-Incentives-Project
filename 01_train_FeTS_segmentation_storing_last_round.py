# ─────────────────────────────────────────────────────────────
#  Imports
# ─────────────────────────────────────────────────────────────
import os
import copy
import time
import glob
import shutil
import tempfile
from itertools import chain, combinations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.special import comb
import matplotlib.pyplot as plt
from tqdm import tqdm
import nibabel as nib
import pulp
import onnxruntime
import random

# ─────────────────────────────────────────────────────────────
#  MONAI
# ─────────────────────────────────────────────────────────────
from monai.config import print_config
from monai.utils import set_determinism
from monai.data import CacheDataset, Dataset, DataLoader, decollate_batch
from monai.handlers.utils import from_engine
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.networks.nets import SegResNet
from monai.apps import DecathlonDataset
from monai.transforms import (
    Activations,
    Activationsd,
    AsDiscrete,
    AsDiscreted,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    Invertd,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    ScaleIntensityd,
    Spacingd,
    SelectItemsd
)

# ─────────────────────────────────────────────────────────────
#  Custom Modules
# ─────────────────────────────────────────────────────────────
from utils import *

# ─────────────────────────────────────────────────────────────
#  Device & Setup
# ─────────────────────────────────────────────────────────────
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print_config()
set_determinism(seed=0)


# Corrected conversion for FeTS labels
class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    FeTS/BraTS label mapping (ints on disk): 0=background, 1=NCR/NET, 2=edema, 4=enhancing (ET)
    Build 3-channel multi-label [TC, WT, ET]:
      TC = (label==1) OR (label==4)
      WT = (label==1) OR (label==2) OR (label==4)
      ET = (label==4)
    """
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            lab = d[key]
            tc = torch.logical_or(lab == 1, lab == 4)
            wt = torch.logical_or(torch.logical_or(lab == 1, lab == 2), lab == 4)
            et = (lab == 4)
            d[key] = torch.stack([tc, wt, et], dim=0).float()
        return d


train_transform = Compose(
    [
        # load 4 Nifti images and stack them together
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys="image"),
        EnsureTyped(keys=["image", "label"]),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        RandSpatialCropd(keys=["image", "label"], roi_size=[224, 224, 144], random_size=False),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
    ]
)
val_transform = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys="image"),
        EnsureTyped(keys=["image", "label"]),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ]
)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------
# 0. paths & meta-data (unchanged) ---------------------------
# -----------------------------------------------------------
BRATS_DIR = "/mnt/d/Datasets/FETS_data/MICCAI_FeTS2022_TrainingData"
CSV_PATH  = f"{BRATS_DIR}/partitioning_1.csv"
MODALITIES = ["flair", "t1", "t1ce", "t2"]
LABEL_KEY  = "seg"

# -----------------------------------------------------------
# 1. read partition file  ➜  { id : [subjects] } ------------
# -----------------------------------------------------------
part_df = pd.read_csv(CSV_PATH)

# --- compute subject counts per site -----------------------
site_counts = (
    part_df.groupby("Partition_ID")["Subject_ID"]
           .nunique()
)

TOP_K = 6  # keep 6 most populated sites for training

# site IDs for training (top-K by subject count)
TRAIN_CENTRES = set(
    site_counts.sort_values(ascending=False)
               .head(TOP_K)
               .index.tolist()
)

# everything else is validation
VAL_CENTRES = set(site_counts.index) - TRAIN_CENTRES

print("Train centres (top 6 by subject count):")
print(site_counts.loc[sorted(TRAIN_CENTRES)])
print("\nValidation centres (remaining):")
print(site_counts.loc[sorted(VAL_CENTRES)])

# map centre → list of subject IDs
partition_map = (
    part_df.groupby("Partition_ID")["Subject_ID"]
           .apply(list).to_dict()
)

# split once, reuse everywhere
train_partitions = {
    cid: sids for cid, sids in partition_map.items()
    if cid in TRAIN_CENTRES
}
val_subjects = sum((partition_map[cid] for cid in VAL_CENTRES), [])


# -----------------------------------------------------------
# 2. helper to build MONAI-style record dicts ----------------
# -----------------------------------------------------------
def build_records(subject_ids):
    recs = []
    for sid in subject_ids:
        sdir = f"{BRATS_DIR}/{sid}"
        images = [f"{sdir}/{sid}_{m}.nii.gz" for m in MODALITIES]  # 4 modalities
        recs.append({"image": images, "label": f"{sdir}/{sid}_{LABEL_KEY}.nii.gz"})
    return recs


# -----------------------------------------------------------
# 3. MONAI CacheDatasets ------------------------------------
# -----------------------------------------------------------
FRAC, SEED = 1.0, 42   # FRAC for subsampling within each site
rng = random.Random(SEED)

train_datasets = {}
for cid, subj_ids in train_partitions.items():
    k = max(1, int(len(subj_ids) * FRAC))   # e.g. 0.3 for 30% subsample
    sample_ids = rng.sample(subj_ids, k)
    train_datasets[cid] = CacheDataset(
        build_records(sample_ids), transform=train_transform, cache_rate=0.1
    )

# ── single validation dataset made from *all* val subjects ─
val_dataset = CacheDataset(
    data=build_records(val_subjects),
    transform=val_transform,
    cache_rate=0.1
)


print("train per-centre sizes:", {k: len(v) for k, v in train_datasets.items()})
print("validation size:", len(val_dataset))


# --- Count train samples per center ---
train_counts = {cid: len(ds) for cid, ds in train_datasets.items()}

# --- Count validation samples per center ---
val_counts = {cid: len(partition_map[cid]) for cid in VAL_CENTRES}

# --- Build DataFrame ---

train_df = pd.DataFrame({
    "centre": sorted(train_datasets.keys()),
    "train_size": [len(train_datasets[cid]) for cid in sorted(train_datasets.keys())]
}).set_index("centre")

print(train_df)

val_df = pd.DataFrame({
    "centre": sorted(VAL_CENTRES),
    "val_size": [len(partition_map[cid]) for cid in sorted(VAL_CENTRES)]
}).set_index("centre")

print(val_df)

max_epochs = 300
val_interval = 1
VAL_AMP = True

# create SegResNet, DiceLoss and Adam optimizer
device = torch.device("cuda:0")
global_model = SegResNet(
    blocks_down=[1, 2, 2, 4],
    blocks_up=[1, 1, 1],
    init_filters=16,
    in_channels=4,
    out_channels=3,
    dropout_prob=0.2,
).to(device)
loss_function = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
optimizer = torch.optim.Adam(global_model.parameters(), 1e-4, weight_decay=1e-5)

dice_metric = DiceMetric(include_background=True, reduction="mean")
dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

# use amp to accelerate training
scaler = torch.GradScaler("cuda")
# enable cuDNN benchmark
torch.backends.cudnn.benchmark = True


def evaluate_model(model, dataset, device, batch_size=1,
                   roi_size=(128, 128, 64), sw_batch_size=4):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    dice_metric.reset()
    dice_metric_batch.reset()
    model.eval()

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            inputs = batch["image"].to(device)
            labels = batch["label"].to(device)          # [B, 3, D, H, W]

            logits = sliding_window_inference(
                inputs=inputs,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                predictor=model,                        # ← use THIS model
            )

            preds = torch.sigmoid(logits)
            preds = (preds > 0.5).float()

            dice_metric(y_pred=preds, y=labels)
            dice_metric_batch(y_pred=preds, y=labels)

    mean_dice = dice_metric.aggregate().item()
    metric_batch = dice_metric_batch.aggregate()
    metric_tc = metric_batch[0].item()
    metric_wt = metric_batch[1].item()
    metric_et = metric_batch[2].item()
    dice_metric.reset()
    dice_metric_batch.reset()

    return mean_dice, metric_tc, metric_wt, metric_et
    
# ─────────────────────────────────────────────────────────────
#  Federation setup
# ─────────────────────────────────────────────────────────────
# train_datasets: dict[int -> MONAI CacheDataset]  (already built)
idxs_users = list(sorted(train_datasets.keys()))
N = len(idxs_users)
print(f"We got {N} clients")

# Fed hyperparams (align with your working pipeline)
ROUNDS       = 100            # you can raise later (e.g., 100)
LOCAL_EPOCHS = 1
LR           = 1e-4
BATCH        = 1

# Client sizes & FedAvg fractions
sizes     = {k: len(ds) for k, ds in train_datasets.items()}
total_n   = sum(sizes.values())
fractions = [sizes[k] / total_n for k in idxs_users]

# Where to persist submodels / global snapshots
submodel_dir = "submodels"
os.makedirs(submodel_dir, exist_ok=True)
submodel_file_template = os.path.join(submodel_dir, "submodel_{}.pth")
global_model_path      = os.path.join(submodel_dir, "global_model.pth")
best_model_path        = os.path.join(submodel_dir, "best_metric_model.pth")

# Save initial global (round 0) – useful for baselines
torch.save(global_model.state_dict(), global_model_path)

# For later Shapley steps
accuracy_dict = {}     # coalition -> utility (e.g., Dice on test set)
shapley_dict  = {}     # client -> shapley value (to be filled later)

# fast sanity check before any training
print("Dice before any training:", evaluate_model(global_model, val_dataset, device))


from tqdm.auto import tqdm, trange   # trange == tqdm(range())
from collections import OrderedDict

def average_weights(state_dicts, fractions):
    """
    Federated averaging with client fractions (must sum to 1).
    state_dicts: list of state_dicts (same keys)
    fractions:   list of floats, same length, sum≈1
    """
    avg_sd = OrderedDict()
    for k in state_dicts[0].keys():
        avg = 0.0
        for sd, w in zip(state_dicts, fractions):
            avg += sd[k] * w
        avg_sd[k] = avg
    return avg_sd

# ────────────────────────────────────────────────────────────
# 1. one-client update (returns weights + mean loss)          │
# ────────────────────────────────────────────────────────────
def local_train(model, loader, device, lr=1e-4, epochs=1):
    """
    Train a local copy of the global model on one client's DataLoader.
    Uses your DiceLoss (multi-label, sigmoid) and full crops from transforms.
    """
    model = copy.deepcopy(model).to(device)
    model.train()

    # reuse your loss choice; or inline DiceLoss the same way
    crit = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True,
                    to_onehot_y=False, sigmoid=True).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    epoch_losses = []

    for _ in range(epochs):
        running = 0.0
        for batch in loader:
            img = batch["image"].to(device)   # [B, 4, D, H, W]
            msk = batch["label"].to(device)   # [B, 3, D, H, W]

            opt.zero_grad(set_to_none=True)
            logits = model(img)               # [B, 3, D, H, W]
            loss = crit(logits, msk)
            loss.backward()
            opt.step()

            running += float(loss.item())
        epoch_losses.append(running / max(1, len(loader)))

    return model.state_dict(), float(np.mean(epoch_losses))


# ─────────────────────────────────────────────────────────────
#  FedAvg training loop (with per-client snapshots each round)
# ─────────────────────────────────────────────────────────────
from tqdm.auto import tqdm, trange
from collections import OrderedDict

best_metric = -1
best_metric_round = -1
best_metrics_rounds_and_time = [[], [], []]   # best, round, seconds
round_loss_values = []
metric_values     = []
metric_values_tc  = []
metric_values_wt  = []
metric_values_et  = []

patience      = 10      # stop after 5 rounds with no improvement
no_improve    = 0
start_time    = time.time()
last_round_run = 0     # track actual last round (for logging)

for rnd in trange(1, ROUNDS + 1, desc="Global rounds", position=0, leave=True, dynamic_ncols=True):
    local_weights, client_losses = [], []

    # —— local updates per client ——
    for cid in tqdm(idxs_users, desc=" clients", position=1, leave=False, total=len(idxs_users), dynamic_ncols=True):
        loader = DataLoader(
            train_datasets[cid], batch_size=BATCH, shuffle=True,
            num_workers=4, pin_memory=True
        )
        w, loss = local_train(global_model, loader, device, lr=LR, epochs=LOCAL_EPOCHS)
        local_weights.append(w); client_losses.append(loss)

        # Persist this client's *latest* local model for Shapley / ablations
        torch.save(w, submodel_file_template.format(cid))

    # —— FedAvg (fraction-weighted) ——
    global_model.load_state_dict(average_weights(local_weights, fractions))

    # —— validation metrics on your current pipeline ——
    mean_dice, metric_tc, metric_wt, metric_et = evaluate_model(global_model, val_dataset, device)
    metric_values.append(mean_dice)
    metric_values_tc.append(metric_tc)
    metric_values_wt.append(metric_wt)
    metric_values_et.append(metric_et)

    mean_loss = float(np.mean(client_losses))
    round_loss_values.append(mean_loss)

    # —— track best & save ——
    if mean_dice > best_metric:
        best_metric = mean_dice
        best_metric_round = rnd
        best_metrics_rounds_and_time[0].append(best_metric)
        best_metrics_rounds_and_time[1].append(best_metric_round)
        best_metrics_rounds_and_time[2].append(time.time() - start_time)
        torch.save(global_model.state_dict(), best_model_path)
        print("saved new best metric model")
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"Early stopping triggered at round {rnd} (no improvement for {patience} rounds).")
            break

    tqdm.write(
        f"Round {rnd:02d}: mean-loss={mean_loss:.4f} "
        f"mean-Dice={mean_dice:.4f}  "
        f"TC-Dice={metric_tc:.4f}  WT-Dice={metric_wt:.4f}  ET-Dice={metric_et:.4f}"
    )

# ── final val utility for the “grand coalition” (all clients) ─────────────
val_mean_dice, val_tc, val_wt, val_et = evaluate_model(global_model, val_dataset, device)
print(f"\nResults after {ROUNDS} global rounds:")
print(f"|---- Val Dice(mean): {val_mean_dice:.4f} | TC {val_tc:.4f} | WT {val_wt:.4f} | ET {val_et:.4f}")

# Store utility for coalition = all clients (tuple keeps order deterministic)
accuracy_dict[tuple(idxs_users)] = val_mean_dice


# below is the shapley calculation. In case training finishes and you're sleeping, it can move on with the next step. If it crashes but previous steps are complete, continue using script #02

from shapley_eval import run_shapley_eval

accuracy_dict, shapley_dict, lc_dict = run_shapley_eval(
    global_model=global_model,
    val_dataset=val_dataset,
    idxs_users=idxs_users,                      # same list you used in training
    fractions=fractions,                        # same FedAvg fractions as training
    submodel_file_template=submodel_file_template,  # e.g. "submodels/submodel_{}.pth"
    device=device,
    coalition_csv="coalition_utilities.csv",
    allocation_csv="allocation_summary.csv",
)

