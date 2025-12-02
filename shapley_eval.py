#!/usr/bin/env python3
"""
Post-hoc coalition evaluation + Shapley + least-core for FL segmentation.

Usage from a notebook (after rebuilding datasets/model):

    from shapley_eval import run_shapley_eval

    accuracy_dict, shapley_dict, lc_dict = run_shapley_eval(
        global_model=global_model,
        val_dataset=val_dataset,
        idxs_users=idxs_users,
        fractions=fractions,
        submodel_file_template=submodel_file_template,
        device=device,
        coalition_csv="coalition_utilities.csv",
        allocation_csv="allocation_summary.csv",
    )
"""

import os
import copy
import time
from itertools import combinations
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from utils import shapley, least_core  # your existing functions
# if you want; you can also import powersettool, etc., from utils


def average_weights(state_dicts, fractions):
    """
    Federated averaging with client fractions (must sum to 1).
    state_dicts: list of state_dicts (same keys)
    fractions:   list of floats, same length, sum≈1
    """
    from collections import OrderedDict
    avg_sd = OrderedDict()
    for k in state_dicts[0].keys():
        avg = 0.0
        for sd, w in zip(state_dicts, fractions):
            avg += sd[k] * w
        avg_sd[k] = avg
    return avg_sd


def subset_weights_and_fracs(
    subset: Tuple[int, ...],
    fraction_map: Dict[int, float],
    submodel_file_template: str,
):
    """
    Load client submodels for this subset and renormalize fractions.
    """
    w_list = [torch.load(submodel_file_template.format(cid), map_location="cpu")
              for cid in subset]
    raw_fracs = np.array([fraction_map[cid] for cid in subset], dtype=float)
    raw_sum = float(raw_fracs.sum())
    if raw_sum <= 0:
        norm_fracs = [1.0 / len(subset)] * len(subset)
    else:
        norm_fracs = (raw_fracs / raw_sum).tolist()
    return w_list, norm_fracs


def stats(vector: np.ndarray, label: str = ""):
    n = len(vector)
    egal = np.array([1 / n for _ in range(n)])
    normalised = np.array(vector / vector.sum())
    msg = ""
    if label:
        msg += f"=== {label} ===\n"
    msg += f"Original vector:   {vector}\n"
    msg += f"Normalised vector: {normalised}\n"
    msg += f"Max Diff:          {normalised.max() - normalised.min():.4f}\n"
    msg += f"Distance:          {np.linalg.norm(normalised - egal):.4f}\n"
    msg += f"Budget (sum):      {vector.sum():.4f}\n"
    print(msg)

def compute_coalition_utilities(
    global_model,
    val_dataset,
    idxs_users: List[int],
    fractions: List[float],
    submodel_file_template: str,
    device: torch.device,
    coalition_csv: str = "./logs/coalition_utilities.csv",
    save_every: int = 1,
):
    """
    Evaluate all coalitions (post-hoc) and save their utilities to CSV.
    This is the expensive, resumable part.

    Parameters
    ----------
    global_model : nn.Module
        Segmentation model with the correct architecture (weights will be overwritten per coalition).
    val_dataset : MONAI Dataset
        Validation dataset (same as used during training).
    idxs_users : list[int]
        List of client IDs (e.g., [1,2,3]).
    fractions : list[float]
        FedAvg fractions per client, in the same order as idxs_users.
    submodel_file_template : str
        Template path for client submodels, e.g. "submodels/submodel_{}.pth".
    device : torch.device
        Device for evaluation.
    coalition_csv : str
        Path to save coalition utilities CSV.
    save_every : int
        How often (in number of newly evaluated coalitions) to checkpoint to CSV.

    Returns
    -------
    accuracy_dict : dict[tuple[int], float]
        Mapping from coalition (tuple of client IDs) to mean validation Dice.
    """

    from monai.data import DataLoader
    from monai.metrics import DiceMetric
    from monai.inferers import sliding_window_inference

    dice_metric = DiceMetric(include_background=True, reduction="mean")
    dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

    def evaluate_model(model, dataset, device, batch_size=1,
                       roi_size=(128, 128, 64), sw_batch_size=4):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
        dice_metric.reset()
        dice_metric_batch.reset()
        model.eval()

        with torch.no_grad():
            for batch in tqdm(loader, desc="Evaluating", leave=False):
                inputs = batch["image"].to(device)
                labels = batch["label"].to(device)
                logits = sliding_window_inference(
                    inputs=inputs,
                    roi_size=roi_size,
                    sw_batch_size=sw_batch_size,
                    predictor=model,
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

    # ── deterministic powerset over your sorted client IDs ──────────────────
    clients = list(sorted(idxs_users))   # e.g., [1,2,3]
    powerset = [tuple(s) for r in range(1, len(clients) + 1)
                for s in combinations(clients, r)]  # non-empty only

    # Build a client -> fraction map (global FedAvg fractions you already computed)
    fraction_map = {cid: frac for cid, frac in zip(clients, fractions)}

    accuracy_dict: Dict[Tuple[int, ...], float] = {}
    coalition_rows = []

    # ── resume from existing coalition CSV if it exists ─────────────────────
    done_coalitions = set()
    if os.path.exists(coalition_csv):
        prev_df = pd.read_csv(coalition_csv)
        for _, row in prev_df.iterrows():
            coalition_tuple = tuple(
                int(x) for x in str(row["coalition"]).split(",") if x != ""
            )
            done_coalitions.add(coalition_tuple)
            accuracy_dict[coalition_tuple] = float(row["val_mean_dice"])
            coalition_rows.append(row.to_dict())
        print(f"[Resume] Loaded {len(done_coalitions)} coalitions from {coalition_csv}")

    start_all = time.time()
    newly_evaluated = 0

    # ── evaluate every proper coalition (exclude grand coalition at first) ──
    for subset in powerset[:-1]:
        if subset in done_coalitions:
            print(f"Skipping coalition {subset} (already evaluated).")
            continue

        if len(subset) == 1:
            subset_sd = torch.load(submodel_file_template.format(subset[0]),
                                   map_location="cpu")
        else:
            w_list, norm_fracs = subset_weights_and_fracs(
                subset, fraction_map, submodel_file_template
            )
            subset_sd = average_weights(w_list, norm_fracs)

        submodel = copy.deepcopy(global_model).to(device)
        submodel.load_state_dict(subset_sd)
        submodel.eval()

        mean_dice, metric_tc, metric_wt, metric_et = evaluate_model(
            submodel, val_dataset, device
        )

        accuracy_dict[subset] = float(mean_dice)
        coalition_rows.append({
            "coalition": ",".join(map(str, subset)),
            "size": len(subset),
            "val_mean_dice": mean_dice,
            "val_tc": metric_tc,
            "val_wt": metric_wt,
            "val_et": metric_et,
        })

        print(
            f"\nCoalition {subset}: "
            f"mean Val Dice={mean_dice:.4f} | "
            f"TC={metric_tc:.4f} | WT={metric_wt:.4f} | ET={metric_et:.4f}"
        )

        newly_evaluated += 1
        if save_every > 0 and newly_evaluated % save_every == 0:
            tmp_path = coalition_csv + ".tmp"
            coalition_df = pd.DataFrame(coalition_rows)
            coalition_df.to_csv(tmp_path, index=False)
            os.replace(tmp_path, coalition_csv)
            print(f"[Checkpoint] Saved {len(coalition_rows)} coalition rows → {coalition_csv}")

        del submodel
        torch.cuda.empty_cache()

    # ── ensure grand coalition utility ──────────────────────────────────────
    grand = tuple(clients)
    if grand not in accuracy_dict:
        print("Evaluating grand coalition...")
        mean_dice, metric_tc, metric_wt, metric_et = evaluate_model(
            global_model.to(device), val_dataset, device
        )
        accuracy_dict[grand] = float(mean_dice)
        coalition_rows.append({
            "coalition": ",".join(map(str, grand)),
            "size": len(grand),
            "val_mean_dice": mean_dice,
            "val_tc": metric_tc,
            "val_wt": metric_wt,
            "val_et": metric_et,
        })
    else:
        print("Grand coalition already present in cache.")

    total_val_time = time.time() - start_all
    print(f" Total coalition eval time: {total_val_time:0.4f}s")
    print(f" Grand-coalition validation utility (Dice): {accuracy_dict[grand]:.4f}")

    # final save
    coalition_df = pd.DataFrame(coalition_rows)
    coalition_df.to_csv(coalition_csv, index=False)
    print(f"\n[Saved] Coalition utilities → {coalition_csv}")

    # add empty coalition utility here as well
    accuracy_dict[()] = 0.0

    return accuracy_dict

def run_shapley_from_coalitions(
    coalition_csv: str = "./logs/coalition_utilities.csv",
    allocation_csv: str = "./logs/allocation_summary.csv",
):
    """
    Load coalition utilities from CSV, reconstruct accuracy_dict (with original
    client labels), then compute Shapley and Least Core allocations by internally
    remapping client IDs to 1..N for the cooperative-game routines. <-- !!! reason for this is because Vasilis' code expects 1 to N.

    Parameters
    ----------
    coalition_csv : str
        Path to coalition utilities CSV (as written by compute_coalition_utilities).
    allocation_csv : str
        Path to save Shapley + LC allocations CSV.

    Returns
    -------
    accuracy_dict : dict[tuple[int], float]
        Coalition utilities keyed by ORIGINAL client IDs.
    shapley_dict  : dict[int, float]
        Shapley values keyed by ORIGINAL client IDs.
    lc_dict       : pulp LpProblem
        Least-core optimization problem (still indexed internally by 1..N).
    """
    if not os.path.exists(coalition_csv):
        raise FileNotFoundError(f"{coalition_csv} not found.")

    coalition_df = pd.read_csv(coalition_csv)

    # ── reconstruct accuracy_dict with ORIGINAL labels ──────────────────────
    accuracy_dict: Dict[Tuple[int, ...], float] = {}
    all_clients = set()

    for _, row in coalition_df.iterrows():
        coalition_str = str(row["coalition"])
        if coalition_str.strip() == "":
            continue
        coalition = tuple(int(x) for x in coalition_str.split(","))
        all_clients.update(coalition)
        accuracy_dict[coalition] = float(row["val_mean_dice"])

    # infer number of clients from CSV
    clients = sorted(all_clients)      # e.g., [1, 4, 6, 13, 18, 21]
    n_clients = len(clients)

    # add empty coalition if missing
    if () not in accuracy_dict:
        accuracy_dict[()] = 0.0

    # sanity: ensure grand coalition present
    grand = tuple(clients)
    if grand not in accuracy_dict:
        raise ValueError(
            f"Grand coalition {grand} not found in {coalition_csv}. "
            "Did you run compute_coalition_utilities() to completion?"
        )

    # ── build label↔index maps for 1..N reindexing ─────────────────────────
    # internal index: 1..N
    label_to_idx = {cid: i + 1 for i, cid in enumerate(clients)}
    idx_to_label = {i + 1: cid for i, cid in enumerate(clients)}

    # remap coalitions in accuracy_dict to 1..N for shapley/LC routines
    utility_idx: Dict[Tuple[int, ...], float] = {}
    for coal, val in accuracy_dict.items():
        # coal is a tuple of ORIGINAL labels; map to contiguous indices
        coal_idx = tuple(label_to_idx[c] for c in coal)
        utility_idx[coal_idx] = val

    # ── Shapley & Least Core on reindexed game ──────────────────────────────
    shap_start = time.time()
    shapley_idx = shapley(utility_idx, n_clients)   # keys are 1..N
    shapTime = time.time() - shap_start

    lc_start = time.time()
    lc_dict = least_core(utility_idx, n_clients)    # also expects 1..N labels
    LCTime = time.time() - lc_start

    # map Shapley values back to ORIGINAL client IDs
    shapley_dict: Dict[int, float] = {
        idx_to_label[idx]: phi for idx, phi in shapley_idx.items()
    }

    print(f"\n Grand-coalition validation utility (Dice): {accuracy_dict[grand]:.4f}")
    print(f" Total Time Shapley (post-hoc only): {shapTime:0.4f}s")
    print(f" Total Time LC (post-hoc only):      {LCTime:0.4f}s")

    # ── print allocations (in terms of ORIGINAL client IDs) ─────────────────
    print("\nShapley allocation:")
    for cid in sorted(shapley_dict.keys()):
        print(f" client {cid}: {shapley_dict[cid]:.4f}")

    print("\nLeast-Core allocation:")
    lc_values: Dict[int, float] = {}
    for var in lc_dict.variables():
        if var.name.startswith("x("):
            # this 'cid_idx' is 1..N; map back to ORIGINAL label
            cid_idx = int(var.name[2:-1])
            cid = idx_to_label[cid_idx]
            lc_values[cid] = var.value()
            print(f" client {cid}: {var.value():.4f}")
    e_slack = lc_dict.variablesDict()['e'].value()
    print(f" e (slack): {e_slack:.4f}")

    # ── allocation summary CSV (ORIGINAL IDs) ───────────────────────────────
    clients_sorted = sorted(shapley_dict.keys())
    alloc_rows = []
    for cid in clients_sorted:
        alloc_rows.append({
            "client": cid,
            "shapley": shapley_dict[cid],
            "least_core": lc_values.get(cid, np.nan),
        })
    alloc_df = pd.DataFrame(alloc_rows)
    alloc_df.to_csv(allocation_csv, index=False)
    print(f"[Saved] Allocation summary → {allocation_csv}")

    # simple fairness stats
    shap_vec = np.array([shapley_dict[cid] for cid in clients_sorted], dtype=float)
    lc_vec = np.array([
        alloc_df.loc[alloc_df.client == cid, "least_core"].item()
        for cid in clients_sorted
    ], dtype=float)

    stats(shap_vec, label="Shapley")
    stats(lc_vec, label="Least-Core")

    return accuracy_dict, shapley_dict, lc_dict
