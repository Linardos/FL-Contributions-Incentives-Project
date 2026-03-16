#!/usr/bin/env python3
"""
Post-hoc coalition evaluation + Shapley + Least-Core for FL segmentation.

Compatibility notes (relative to the previous collaborator version):
- Default behavior is unchanged: `method="snapshot"` still evaluates coalitions by
  averaging final local client snapshots.
- Two optional methods are now exposed explicitly:
  - `method="or"`: one-round approximation via singleton trajectories.
  - `method="mr"`: multi-round approximation via coalition trajectories.
- Performance knobs are optional and backward-safe:
  - `max_samples` caps validation-set samples for faster profiling runs.
  - `save_every` controls resumable CSV checkpoint cadence.
- CSV outputs are unchanged:
  - coalition CSV: `coalition,size,val_mean_dice,val_tc,val_wt,val_et`
  - allocation CSV: `client,shapley,least_core`
"""

from __future__ import annotations

import copy
import os
import re
import time
from itertools import combinations
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset
from tqdm.auto import tqdm

from utils import least_core, shapley

Coalition = Tuple[int, ...]
StateDict = Mapping[str, torch.Tensor]


def _log_header(title: str) -> None:
    print(f"\n{'=' * 18} {title} {'=' * 18}")


def _canonical_clients(idxs_users: Sequence[int]) -> List[int]:
    clients = sorted(int(c) for c in idxs_users)
    if len(clients) != len(set(clients)):
        raise ValueError(f"Duplicate client IDs in idxs_users: {idxs_users}")
    return clients


def _normalize_fracs(values: Sequence[float]) -> List[float]:
    arr = np.array(values, dtype=float)
    total = float(arr.sum())
    if total <= 0:
        return [1.0 / len(arr) for _ in arr]
    return (arr / total).tolist()


def _sync_running_buffers(target: Dict[str, torch.Tensor], reference: StateDict) -> None:
    """Copy running-stat buffers from a reference state dict."""

    for key in target.keys():
        if "running" in key or "batches" in key or "num_batches_tracked" in key:
            target[key] = reference[key].clone()


def _calculate_gradients(
    new_weights: Sequence[StateDict],
    old_weights: StateDict,
) -> List[Dict[str, torch.Tensor]]:
    gradients: List[Dict[str, torch.Tensor]] = []
    for local_state in new_weights:
        grad: Dict[str, torch.Tensor] = {}
        for key, old_tensor in old_weights.items():
            local_tensor = local_state[key]
            if torch.is_floating_point(old_tensor) or torch.is_complex(old_tensor):
                grad[key] = local_tensor - old_tensor
            else:
                grad[key] = torch.zeros_like(old_tensor)
        gradients.append(grad)
    return gradients


def _update_weights_from_gradients(
    gradients: StateDict,
    old_weights: StateDict,
) -> Dict[str, torch.Tensor]:
    updated: Dict[str, torch.Tensor] = {}
    for key, old_tensor in old_weights.items():
        if torch.is_floating_point(old_tensor) or torch.is_complex(old_tensor):
            updated[key] = old_tensor + gradients[key]
        else:
            updated[key] = old_tensor.clone()
    return updated


def average_weights(state_dicts: Sequence[StateDict], fractions: Sequence[float]) -> Dict[str, torch.Tensor]:
    """Weighted average for state dicts with proper normalization."""

    if len(state_dicts) == 0:
        raise ValueError("state_dicts must be non-empty")
    if len(state_dicts) != len(fractions):
        raise ValueError("state_dicts and fractions must have same length")

    norm_fracs = _normalize_fracs(fractions)
    avg_sd: Dict[str, torch.Tensor] = {}

    for key in state_dicts[0].keys():
        first = state_dicts[0][key]
        if torch.is_floating_point(first) or torch.is_complex(first):
            value = first * float(norm_fracs[0])
            for i in range(1, len(state_dicts)):
                value = value + state_dicts[i][key] * float(norm_fracs[i])
            avg_sd[key] = value
        else:
            avg_sd[key] = first.clone()

    return avg_sd


def subset_weights_and_fracs(
    subset: Coalition,
    fraction_map: Mapping[int, float],
    submodel_file_template: str,
) -> Tuple[List[StateDict], List[float]]:
    """Load client submodels for a subset and normalize subset fractions."""

    w_list = [torch.load(submodel_file_template.format(cid), map_location="cpu") for cid in subset]
    norm_fracs = _normalize_fracs([fraction_map[cid] for cid in subset])
    return w_list, norm_fracs


def stats(vector: np.ndarray, label: str = "") -> None:
    n = len(vector)
    egal = np.array([1 / n for _ in range(n)])
    total = float(vector.sum())
    normalised = np.array(vector / total) if total > 0 else np.array([1 / n for _ in range(n)])
    msg = ""
    if label:
        msg += f"=== {label} ===\n"
    msg += f"Original vector:   {vector}\n"
    msg += f"Normalised vector: {normalised}\n"
    msg += f"Max Diff:          {normalised.max() - normalised.min():.4f}\n"
    msg += f"Distance:          {np.linalg.norm(normalised - egal):.4f}\n"
    msg += f"Budget (sum):      {vector.sum():.4f}\n"
    print(msg)


def _parse_coalition_csv_entry(value: object) -> Coalition:
    txt = str(value).strip()
    if txt == "":
        return ()
    return tuple(int(x) for x in txt.split(",") if x != "")


def _load_existing_coalition_rows(
    coalition_csv: str,
) -> Tuple[Dict[Coalition, float], List[dict], set[Coalition]]:
    accuracy_dict: Dict[Coalition, float] = {}
    coalition_rows: List[dict] = []
    done_coalitions: set[Coalition] = set()

    if not os.path.exists(coalition_csv):
        return accuracy_dict, coalition_rows, done_coalitions

    prev_df = pd.read_csv(coalition_csv)
    for _, row in prev_df.iterrows():
        coalition_tuple = _parse_coalition_csv_entry(row["coalition"])
        done_coalitions.add(coalition_tuple)
        accuracy_dict[coalition_tuple] = float(row["val_mean_dice"])
        coalition_rows.append(row.to_dict())

    print(f"[Resume] Loaded {len(done_coalitions)} coalitions from {coalition_csv}")
    return accuracy_dict, coalition_rows, done_coalitions


def _save_coalition_rows(coalition_csv: str, coalition_rows: Sequence[dict]) -> None:
    tmp_path = coalition_csv + ".tmp"
    coalition_df = pd.DataFrame(coalition_rows)
    coalition_df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, coalition_csv)


def _build_eval_loader(dataset: object, max_samples: int | None) -> object:
    if max_samples is None:
        return dataset
    if max_samples <= 0:
        raise ValueError("max_samples must be positive when provided")
    if len(dataset) <= max_samples:
        return dataset
    return Subset(dataset, list(range(max_samples)))


def _make_segmentation_evaluator(
    val_dataset: object,
    device: torch.device,
    *,
    batch_size: int,
    roi_size: Tuple[int, int, int],
    sw_batch_size: int,
    num_workers: int,
    pin_memory: bool,
    use_amp: bool,
    show_batch_progress: bool,
    max_samples: int | None,
):
    from monai.data import DataLoader
    from monai.inferers import sliding_window_inference
    from monai.metrics import DiceMetric

    eval_dataset = _build_eval_loader(val_dataset, max_samples=max_samples)
    loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

    def evaluate(model: torch.nn.Module) -> Tuple[float, float, float, float]:
        dice_metric.reset()
        dice_metric_batch.reset()
        model.eval()

        amp_enabled = bool(use_amp and device.type == "cuda")
        with torch.inference_mode():
            iterator = loader
            if show_batch_progress:
                iterator = tqdm(loader, desc="Evaluating", leave=False)
            for batch in iterator:
                inputs = batch["image"].to(device)
                labels = batch["label"].to(device)
                if amp_enabled:
                    with torch.autocast(device_type="cuda", enabled=True):
                        logits = sliding_window_inference(
                            inputs=inputs,
                            roi_size=roi_size,
                            sw_batch_size=sw_batch_size,
                            predictor=model,
                        )
                else:
                    logits = sliding_window_inference(
                        inputs=inputs,
                        roi_size=roi_size,
                        sw_batch_size=sw_batch_size,
                        predictor=model,
                    )
                preds = (torch.sigmoid(logits) > 0.5).float()
                dice_metric(y_pred=preds, y=labels)
                dice_metric_batch(y_pred=preds, y=labels)

        mean_dice = float(dice_metric.aggregate().item())
        metric_batch = dice_metric_batch.aggregate()
        metric_tc = float(metric_batch[0].item())
        metric_wt = float(metric_batch[1].item())
        metric_et = float(metric_batch[2].item())
        dice_metric.reset()
        dice_metric_batch.reset()
        return mean_dice, metric_tc, metric_wt, metric_et

    return evaluate, len(eval_dataset)


def _coalitions(clients: Sequence[int]) -> List[Coalition]:
    return [tuple(s) for r in range(1, len(clients) + 1) for s in combinations(clients, r)]


def _discover_rounds(round_artifacts_dir: str) -> List[int]:
    if not os.path.isdir(round_artifacts_dir):
        raise FileNotFoundError(f"Round artifacts dir not found: {round_artifacts_dir}")

    pattern = re.compile(r"round_(\d+)_global_start\.pth$")
    rounds: List[int] = []
    for filename in os.listdir(round_artifacts_dir):
        match = pattern.match(filename)
        if match:
            rounds.append(int(match.group(1)))

    rounds = sorted(set(rounds))
    if not rounds:
        raise FileNotFoundError(
            "No round global checkpoints found. Expected files like "
            "round_0001_global_start.pth"
        )
    return rounds


def _round_global_path(round_artifacts_dir: str, round_idx: int) -> str:
    return os.path.join(round_artifacts_dir, f"round_{round_idx:04d}_global_start.pth")


def _round_local_path(round_artifacts_dir: str, round_idx: int, client_id: int) -> str:
    return os.path.join(round_artifacts_dir, f"round_{round_idx:04d}_client_{client_id}.pth")


def _load_round_or_singletons(
    clients: Sequence[int],
    fractions: Sequence[float],
    round_artifacts_dir: str,
    *,
    sync_running_buffers: bool,
    progress: bool,
) -> Dict[int, Dict[str, torch.Tensor]]:
    round_indices = _discover_rounds(round_artifacts_dir)
    if len(clients) != len(fractions):
        raise ValueError("clients/fractions mismatch while building OR trajectories")

    singleton_states: Dict[int, Dict[str, torch.Tensor]] = {}
    iterator = tqdm(round_indices, desc="OR rounds", leave=False) if progress else round_indices

    for round_idx in iterator:
        global_start = torch.load(_round_global_path(round_artifacts_dir, round_idx), map_location="cpu")
        local_states = [
            torch.load(_round_local_path(round_artifacts_dir, round_idx, cid), map_location="cpu")
            for cid in clients
        ]

        if not singleton_states:
            for cid in clients:
                singleton_states[cid] = copy.deepcopy(global_start)

        gradients = _calculate_gradients(local_states, global_start)
        for cid, grad in zip(clients, gradients):
            updated = _update_weights_from_gradients(grad, singleton_states[cid])
            if sync_running_buffers:
                _sync_running_buffers(updated, global_start)
            singleton_states[cid] = updated

    return singleton_states


def _load_round_mr_submodels(
    clients: Sequence[int],
    fractions: Sequence[float],
    round_artifacts_dir: str,
    *,
    sync_running_buffers: bool,
    progress: bool,
    max_mr_coalitions: int,
) -> Dict[Coalition, Dict[str, torch.Tensor]]:
    round_indices = _discover_rounds(round_artifacts_dir)
    powerset = _coalitions(clients)
    grand = tuple(clients)
    proper_nonempty = [subset for subset in powerset if subset != grand]

    if len(proper_nonempty) > max_mr_coalitions:
        raise ValueError(
            f"MR requested with {len(proper_nonempty)} tracked coalitions (> max_mr_coalitions={max_mr_coalitions}). "
            "Increase max_mr_coalitions if this is intentional."
        )

    fraction_map = {cid: frac for cid, frac in zip(clients, fractions)}
    initial_global = torch.load(_round_global_path(round_artifacts_dir, round_indices[0]), map_location="cpu")

    subset_states: Dict[Coalition, Dict[str, torch.Tensor]] = {
        subset: copy.deepcopy(initial_global) for subset in proper_nonempty
    }

    iterator = tqdm(round_indices, desc="MR rounds", leave=False) if progress else round_indices

    for round_idx in iterator:
        global_start = torch.load(_round_global_path(round_artifacts_dir, round_idx), map_location="cpu")
        local_states = {
            cid: torch.load(_round_local_path(round_artifacts_dir, round_idx, cid), map_location="cpu")
            for cid in clients
        }
        gradients = _calculate_gradients([local_states[cid] for cid in clients], global_start)
        gradient_map = {cid: grad for cid, grad in zip(clients, gradients)}

        for subset in proper_nonempty:
            subset_grads = [gradient_map[cid] for cid in subset]
            subset_fracs = _normalize_fracs([fraction_map[cid] for cid in subset])
            subset_gradient = average_weights(subset_grads, subset_fracs)
            updated = _update_weights_from_gradients(subset_gradient, subset_states[subset])
            if sync_running_buffers:
                _sync_running_buffers(updated, global_start)
            subset_states[subset] = updated

    return subset_states


def _format_row(subset: Coalition, mean_dice: float, tc: float, wt: float, et: float) -> dict:
    return {
        "coalition": ",".join(map(str, subset)),
        "size": len(subset),
        "val_mean_dice": mean_dice,
        "val_tc": tc,
        "val_wt": wt,
        "val_et": et,
    }


def compute_coalition_utilities(
    global_model,
    val_dataset,
    idxs_users: List[int],
    fractions: List[float],
    submodel_file_template: str,
    device: torch.device,
    coalition_csv: str = "./logs/coalition_utilities.csv",
    save_every: int = 25,
    *,
    method: str = "snapshot",
    round_artifacts_dir: str | None = None,
    max_samples: int | None = None,
    batch_size: int = 1,
    roi_size: Tuple[int, int, int] = (128, 128, 64),
    sw_batch_size: int = 4,
    num_workers: int = 4,
    pin_memory: bool = True,
    use_amp: bool = False,
    progress: bool = True,
    show_batch_progress: bool = False,
    sync_running_buffers: bool = True,
    max_mr_coalitions: int = 4096,
):
    """Evaluate coalition utilities and persist to CSV.

    Parameters
    ----------
    method:
        "snapshot" = current collaborator workflow (average final local snapshots)
        "or"       = singleton trajectory approximation (original OR spirit)
        "mr"       = coalition trajectory approximation (multi-round style)
    round_artifacts_dir:
        Required for method in {"or", "mr"}. Expected files per round:
          - round_XXXX_global_start.pth
          - round_XXXX_client_<cid>.pth
    max_samples:
        Optional cap on validation samples for faster profiling runs.
    """

    method = method.lower().strip()
    if method not in {"snapshot", "or", "mr"}:
        raise ValueError(f"Unsupported method '{method}'. Use one of: snapshot, or, mr")

    clients = _canonical_clients(idxs_users)
    if len(clients) != len(fractions):
        raise ValueError("idxs_users and fractions must have the same length")

    fraction_map = {cid: frac for cid, frac in zip(clients, fractions)}
    powerset = _coalitions(clients)
    grand = tuple(clients)

    accuracy_dict, coalition_rows, done_coalitions = _load_existing_coalition_rows(coalition_csv)

    required = set(powerset)
    if required.issubset(done_coalitions):
        print("[Resume] Coalition cache already complete. Skipping coalition evaluation.")
        accuracy_dict[()] = 0.0
        return accuracy_dict

    evaluate_model, eval_len = _make_segmentation_evaluator(
        val_dataset,
        device,
        batch_size=batch_size,
        roi_size=roi_size,
        sw_batch_size=sw_batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        use_amp=use_amp,
        show_batch_progress=show_batch_progress,
        max_samples=max_samples,
    )

    _log_header("Coalition Utility Evaluation")
    print(f"method={method} | clients={len(clients)} | evaluated_samples={eval_len} | save_every={save_every}")

    start_all = time.time()
    newly_evaluated = 0

    eval_model = copy.deepcopy(global_model).to(device)

    # --- prepare subset model provider -------------------------------------------------
    snapshot_cache: Dict[int, StateDict] = {}

    def get_snapshot_state(cid: int) -> StateDict:
        if cid not in snapshot_cache:
            snapshot_cache[cid] = torch.load(submodel_file_template.format(cid), map_location="cpu")
        return snapshot_cache[cid]

    singleton_states: Dict[int, Dict[str, torch.Tensor]] | None = None
    mr_states: Dict[Coalition, Dict[str, torch.Tensor]] | None = None

    if method == "or":
        if round_artifacts_dir is None:
            raise ValueError("round_artifacts_dir is required for method='or'")
        singleton_states = _load_round_or_singletons(
            clients,
            fractions,
            round_artifacts_dir,
            sync_running_buffers=sync_running_buffers,
            progress=progress,
        )
    elif method == "mr":
        if round_artifacts_dir is None:
            raise ValueError("round_artifacts_dir is required for method='mr'")
        mr_states = _load_round_mr_submodels(
            clients,
            fractions,
            round_artifacts_dir,
            sync_running_buffers=sync_running_buffers,
            progress=progress,
            max_mr_coalitions=max_mr_coalitions,
        )

    iterator: Iterable[Coalition] = powerset[:-1]
    if progress:
        iterator = tqdm(powerset[:-1], desc="Coalitions", leave=True)

    for subset in iterator:
        if subset in done_coalitions:
            continue

        if method == "snapshot":
            if len(subset) == 1:
                subset_sd = get_snapshot_state(subset[0])
            else:
                subset_sd = average_weights(
                    [get_snapshot_state(cid) for cid in subset],
                    _normalize_fracs([fraction_map[cid] for cid in subset]),
                )
        elif method == "or":
            assert singleton_states is not None
            if len(subset) == 1:
                subset_sd = singleton_states[subset[0]]
            else:
                subset_sd = average_weights(
                    [singleton_states[cid] for cid in subset],
                    _normalize_fracs([fraction_map[cid] for cid in subset]),
                )
        else:  # method == "mr"
            assert mr_states is not None
            subset_sd = mr_states[subset]

        eval_model.load_state_dict(subset_sd, strict=True)
        mean_dice, metric_tc, metric_wt, metric_et = evaluate_model(eval_model)

        accuracy_dict[subset] = float(mean_dice)
        coalition_rows.append(_format_row(subset, mean_dice, metric_tc, metric_wt, metric_et))

        newly_evaluated += 1
        if save_every > 0 and newly_evaluated % save_every == 0:
            _save_coalition_rows(coalition_csv, coalition_rows)
            print(f"[Checkpoint] Saved {len(coalition_rows)} coalition rows -> {coalition_csv}")

    if grand not in accuracy_dict:
        print("Evaluating grand coalition...")
        mean_dice, metric_tc, metric_wt, metric_et = evaluate_model(global_model.to(device))
        accuracy_dict[grand] = float(mean_dice)
        coalition_rows.append(_format_row(grand, mean_dice, metric_tc, metric_wt, metric_et))
    else:
        print("Grand coalition already present in cache.")

    _save_coalition_rows(coalition_csv, coalition_rows)

    total_val_time = time.time() - start_all
    print(f"Total coalition eval time: {total_val_time:0.4f}s")
    print(f"Grand-coalition validation utility (Dice): {accuracy_dict[grand]:.4f}")
    print(f"[Saved] Coalition utilities -> {coalition_csv}")

    accuracy_dict[()] = 0.0
    return accuracy_dict


def run_shapley_from_coalitions(
    coalition_csv: str = "./logs/coalition_utilities.csv",
    allocation_csv: str = "./logs/allocation_summary.csv",
):
    """Load coalition utilities from CSV and compute Shapley + Least-Core allocations."""

    if not os.path.exists(coalition_csv):
        raise FileNotFoundError(f"{coalition_csv} not found.")

    coalition_df = pd.read_csv(coalition_csv)

    accuracy_dict: Dict[Coalition, float] = {}
    all_clients = set()
    for _, row in coalition_df.iterrows():
        coalition = _parse_coalition_csv_entry(row["coalition"])
        if coalition == ():
            continue
        all_clients.update(coalition)
        accuracy_dict[coalition] = float(row["val_mean_dice"])

    clients = sorted(all_clients)
    n_clients = len(clients)

    if () not in accuracy_dict:
        accuracy_dict[()] = 0.0

    grand = tuple(clients)
    if grand not in accuracy_dict:
        raise ValueError(
            f"Grand coalition {grand} not found in {coalition_csv}. "
            "Did you run compute_coalition_utilities() to completion?"
        )

    label_to_idx = {cid: i + 1 for i, cid in enumerate(clients)}
    idx_to_label = {i + 1: cid for i, cid in enumerate(clients)}

    utility_idx: Dict[Coalition, float] = {}
    for coal, val in accuracy_dict.items():
        utility_idx[tuple(label_to_idx[c] for c in coal)] = val

    _log_header("Allocation")
    shap_start = time.time()
    shapley_idx = shapley(utility_idx, n_clients)
    shap_time = time.time() - shap_start

    lc_start = time.time()
    lc_dict = least_core(utility_idx, n_clients)
    lc_time = time.time() - lc_start

    shapley_dict: Dict[int, float] = {idx_to_label[idx]: phi for idx, phi in shapley_idx.items()}

    print(f"Grand-coalition validation utility (Dice): {accuracy_dict[grand]:.4f}")
    print(f"Total Time Shapley (post-hoc only): {shap_time:0.4f}s")
    print(f"Total Time LC (post-hoc only):      {lc_time:0.4f}s")

    print("\nShapley allocation:")
    for cid in sorted(shapley_dict.keys()):
        print(f" client {cid}: {shapley_dict[cid]:.4f}")

    print("\nLeast-Core allocation:")
    lc_values: Dict[int, float] = {}
    for var in lc_dict.variables():
        if var.name.startswith("x("):
            cid_idx = int(var.name[2:-1])
            cid = idx_to_label[cid_idx]
            lc_values[cid] = var.value()
            print(f" client {cid}: {var.value():.4f}")
    e_slack = lc_dict.variablesDict()["e"].value()
    print(f" e (slack): {e_slack:.4f}")

    clients_sorted = sorted(shapley_dict.keys())
    alloc_rows = [
        {
            "client": cid,
            "shapley": shapley_dict[cid],
            "least_core": lc_values.get(cid, np.nan),
        }
        for cid in clients_sorted
    ]
    alloc_df = pd.DataFrame(alloc_rows)
    alloc_df.to_csv(allocation_csv, index=False)
    print(f"[Saved] Allocation summary -> {allocation_csv}")

    shap_vec = np.array([shapley_dict[cid] for cid in clients_sorted], dtype=float)
    lc_vec = np.array(
        [alloc_df.loc[alloc_df.client == cid, "least_core"].item() for cid in clients_sorted],
        dtype=float,
    )
    stats(shap_vec, label="Shapley")
    stats(lc_vec, label="Least-Core")

    return accuracy_dict, shapley_dict, lc_dict


def run_shapley_eval(
    global_model,
    val_dataset,
    idxs_users: List[int],
    fractions: List[float],
    submodel_file_template: str,
    device: torch.device,
    coalition_csv: str = "./logs/coalition_utilities.csv",
    allocation_csv: str = "./logs/allocation_summary.csv",
    *,
    method: str = "snapshot",
    round_artifacts_dir: str | None = None,
    max_samples: int | None = None,
    save_every: int = 25,
    batch_size: int = 1,
    roi_size: Tuple[int, int, int] = (128, 128, 64),
    sw_batch_size: int = 4,
    num_workers: int = 4,
    pin_memory: bool = True,
    use_amp: bool = False,
    progress: bool = True,
    show_batch_progress: bool = False,
    sync_running_buffers: bool = True,
    max_mr_coalitions: int = 4096,
):
    """One-call pipeline: coalition utilities + Shapley + Least-Core."""

    compute_coalition_utilities(
        global_model=global_model,
        val_dataset=val_dataset,
        idxs_users=idxs_users,
        fractions=fractions,
        submodel_file_template=submodel_file_template,
        device=device,
        coalition_csv=coalition_csv,
        save_every=save_every,
        method=method,
        round_artifacts_dir=round_artifacts_dir,
        max_samples=max_samples,
        batch_size=batch_size,
        roi_size=roi_size,
        sw_batch_size=sw_batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        use_amp=use_amp,
        progress=progress,
        show_batch_progress=show_batch_progress,
        sync_running_buffers=sync_running_buffers,
        max_mr_coalitions=max_mr_coalitions,
    )

    return run_shapley_from_coalitions(
        coalition_csv=coalition_csv,
        allocation_csv=allocation_csv,
    )
