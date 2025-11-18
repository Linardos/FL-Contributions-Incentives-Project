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


def run_shapley_eval(
    global_model,
    val_dataset,
    idxs_users: List[int],
    fractions: List[float],
    submodel_file_template: str,
    device: torch.device,
    coalition_csv: str = "coalition_utilities.csv",
    allocation_csv: str = "allocation_summary.csv",
):
    """
    Post-hoc evaluation of all coalitions using saved client submodels.

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
    allocation_csv : str
        Path to save Shapley + LC allocations CSV.

    Returns
    -------
    accuracy_dict : dict[tuple[int], float]
    shapley_dict  : dict[int, float]
    lc_dict       : pulp LpProblem
    """
    from monai.data import DataLoader
    from monai.metrics import DiceMetric
    from monai.inferers import sliding_window_inference

    # ----- helper evaluator (uses passed-in model, not any global) ----------
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

    start_all = time.time()

    # ── evaluate every proper coalition (exclude grand coalition at first) ──
    for subset in powerset[:-1]:
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

        del submodel
        torch.cuda.empty_cache()

    # ── ensure grand coalition utility ──────────────────────────────────────
    grand = tuple(clients)
    if grand not in accuracy_dict:
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

    # add empty coalition utility
    accuracy_dict[()] = 0.0

    # ── Shapley & Least Core ────────────────────────────────────────
    shap_start = time.time()
    shapley_dict = shapley(accuracy_dict, len(clients))
    shapTime = time.time() - shap_start

    lc_start = time.time()
    lc_dict = least_core(accuracy_dict, len(clients))
    LCTime = time.time() - lc_start

    total_val_time = time.time() - start_all

    print(f"\n Grand-coalition validation utility (Dice): {accuracy_dict[grand]:.4f}")
    print(f" Total Time Shapley (post-hoc only): {shapTime:0.4f}s")
    print(f" Total Time LC (post-hoc only):      {LCTime:0.4f}s")
    print(f" Total coalition eval time:          {total_val_time:0.4f}s")

    # ── print allocations ───────────────────────────────────────────
    print("\nShapley allocation:")
    for cid, phi in shapley_dict.items():
        print(f" client {cid}: {phi:.4f}")

    print("\nLeast-Core allocation:")
    lc_values = {}
    for var in lc_dict.variables():
        if var.name.startswith("x("):
            cid = int(var.name[2:-1])
            lc_values[cid] = var.value()
            print(f" client {cid}: {var.value():.4f}")
    e_slack = lc_dict.variablesDict()['e'].value()
    print(f" e (slack): {e_slack:.4f}")

    # ── store stats and CSVs ────────────────────────────────────────
    # coalition utilities
    coalition_df = pd.DataFrame(coalition_rows)
    coalition_df.to_csv(coalition_csv, index=False)
    print(f"\n[Saved] Coalition utilities → {coalition_csv}")

    # allocation summary
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
    lc_vec = np.array([alloc_df.loc[alloc_df.client == cid, "least_core"].item()
                       for cid in clients_sorted], dtype=float)

    stats(shap_vec, label="Shapley")
    stats(lc_vec, label="Least-Core")

    return accuracy_dict, shapley_dict, lc_dict
