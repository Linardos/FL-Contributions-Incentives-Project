# ─────────────────────────────────────────────────────────────
#  Imports (minimal for post-hoc Shapley eval)
# ─────────────────────────────────────────────────────────────
import os
import random
from itertools import combinations

import numpy as np
import pandas as pd
import torch

from monai.config import print_config
from monai.utils import set_determinism
from monai.data import CacheDataset
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    NormalizeIntensityd,
    Spacingd,
    MapTransform,
)
from monai.networks.nets import SegResNet

from utils import shapley, least_core  # used inside run_shapley_eval if you keep that design

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print_config()
set_determinism(seed=0)

# ─────────────────────────────────────────────────────────────
#  Label converter (same as training)
# ─────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────
#  Paths & basic meta
# ─────────────────────────────────────────────────────────────
BRATS_DIR = "/mnt/d/Datasets/FETS_data/MICCAI_FeTS2022_TrainingData"
CSV_PATH  = f"{BRATS_DIR}/partitioning_1.csv"
MODALITIES = ["flair", "t1", "t1ce", "t2"]
LABEL_KEY  = "seg"

TOP_K_TRAIN_SITES = 6      # must match training logic
FRAC, SEED = 1.0, 42       # must match training script

# ─────────────────────────────────────────────────────────────
#  Partition map and train/val split (no train CacheDataset)
# ─────────────────────────────────────────────────────────────
part_df = pd.read_csv(CSV_PATH)

# 1) subject counts per site
site_counts = (
    part_df.groupby("Partition_ID")["Subject_ID"]
           .nunique()
)

# 2) pick top-K most populated sites for training
TRAIN_CENTRES = set(
    site_counts.sort_values(ascending=False)
               .head(TOP_K_TRAIN_SITES)
               .index.tolist()
)

# 3) remaining sites are validation
VAL_CENTRES = set(site_counts.index) - TRAIN_CENTRES

print("Train centres (top 6 by subject count):")
print(site_counts.loc[sorted(TRAIN_CENTRES)])
print("\nValidation centres (remaining):")
print(site_counts.loc[sorted(VAL_CENTRES)])

# 4) map centre → subject IDs
partition_map = (
    part_df.groupby("Partition_ID")["Subject_ID"]
           .apply(list).to_dict()
)

train_partitions = {
    cid: sids for cid, sids in partition_map.items()
    if cid in TRAIN_CENTRES
}
val_subjects = sum((partition_map[cid] for cid in VAL_CENTRES), [])

# clients actually used in training
idxs_users = sorted(train_partitions.keys())

# recompute *sizes* exactly as in training: k = max(1, int(len(subj_ids) * FRAC))
sizes = {}
for cid in idxs_users:
    subj_ids = train_partitions[cid]
    k = max(1, int(len(subj_ids) * FRAC))
    sizes[cid] = k

total_n = sum(sizes.values())
fractions = [sizes[cid] / total_n for cid in idxs_users]

print("\nclients:", idxs_users)
print("sizes:", sizes)
print("fractions:", fractions)

# ─────────────────────────────────────────────────────────────
#  Validation dataset only
# ─────────────────────────────────────────────────────────────
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

def build_records(subject_ids):
    recs = []
    for sid in subject_ids:
        sdir = f"{BRATS_DIR}/{sid}"
        images = [f"{sdir}/{sid}_{m}.nii.gz" for m in MODALITIES]  # 4 modalities
        recs.append({"image": images, "label": f"{sdir}/{sid}_{LABEL_KEY}.nii.gz"})
    return recs

val_dataset = CacheDataset(
    build_records(val_subjects), transform=val_transform, cache_rate=1
)
print("validation size:", len(val_dataset))

# ─────────────────────────────────────────────────────────────
#  Global model architecture + loading best weights
# ─────────────────────────────────────────────────────────────
global_model = SegResNet(
    blocks_down=[1, 2, 2, 4],
    blocks_up=[1, 1, 1],
    init_filters=16,
    in_channels=4,
    out_channels=3,
    dropout_prob=0.2,
).to(device)

submodel_dir = "submodels"
submodel_file_template = os.path.join(submodel_dir, "submodel_{}.pth")
best_model_path = os.path.join(submodel_dir, "best_metric_model.pth")

global_model.load_state_dict(torch.load(best_model_path, map_location=device))
global_model.eval()

print("Loaded best global model from:", best_model_path)


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
