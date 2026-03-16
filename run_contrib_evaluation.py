#!/usr/bin/env python3
"""Convenience wrappers for OR/MR contribution evaluation.

Usage from notebook/script:

    from run_contrib_evaluation import run_or, run_mr

    accuracy_dict, shapley_dict, lc_dict = run_or(
        global_model=global_model,
        val_dataset=val_dataset,
        idxs_users=idxs_users,
        fractions=fractions,
        submodel_file_template="submodels/submodel_{}.pth",
        device=device,
        round_artifacts_dir="submodels/round_artifacts",
        coalition_csv="./logs/coalition_utilities.csv",
        allocation_csv="./logs/allocation_summary.csv",
        max_samples=64,
    )
"""

from __future__ import annotations

from typing import List, Tuple

import torch

from shapley_eval import run_shapley_eval


def run_or(
    global_model,
    val_dataset,
    idxs_users: List[int],
    fractions: List[float],
    submodel_file_template: str,
    device: torch.device,
    round_artifacts_dir: str,
    coalition_csv: str = "./logs/coalition_utilities.csv",
    allocation_csv: str = "./logs/allocation_summary.csv",
    *,
    max_samples: int | None = None,
    save_every: int = 25,
    progress: bool = True,
):
    """Run OR-style valuation using round artifacts."""

    return run_shapley_eval(
        global_model=global_model,
        val_dataset=val_dataset,
        idxs_users=idxs_users,
        fractions=fractions,
        submodel_file_template=submodel_file_template,
        device=device,
        coalition_csv=coalition_csv,
        allocation_csv=allocation_csv,
        method="or",
        round_artifacts_dir=round_artifacts_dir,
        max_samples=max_samples,
        save_every=save_every,
        progress=progress,
    )


def run_mr(
    global_model,
    val_dataset,
    idxs_users: List[int],
    fractions: List[float],
    submodel_file_template: str,
    device: torch.device,
    round_artifacts_dir: str,
    coalition_csv: str = "./logs/coalition_utilities.csv",
    allocation_csv: str = "./logs/allocation_summary.csv",
    *,
    max_samples: int | None = None,
    save_every: int = 25,
    progress: bool = True,
    max_mr_coalitions: int = 4096,
):
    """Run MR-style valuation using round artifacts."""

    return run_shapley_eval(
        global_model=global_model,
        val_dataset=val_dataset,
        idxs_users=idxs_users,
        fractions=fractions,
        submodel_file_template=submodel_file_template,
        device=device,
        coalition_csv=coalition_csv,
        allocation_csv=allocation_csv,
        method="mr",
        round_artifacts_dir=round_artifacts_dir,
        max_samples=max_samples,
        save_every=save_every,
        progress=progress,
        max_mr_coalitions=max_mr_coalitions,
    )


def run_snapshot(
    global_model,
    val_dataset,
    idxs_users: List[int],
    fractions: List[float],
    submodel_file_template: str,
    device: torch.device,
    coalition_csv: str = "./logs/coalition_utilities.csv",
    allocation_csv: str = "./logs/allocation_summary.csv",
    *,
    max_samples: int | None = None,
    save_every: int = 25,
    progress: bool = True,
):
    """Run snapshot-style valuation (collaborator's existing workflow)."""

    return run_shapley_eval(
        global_model=global_model,
        val_dataset=val_dataset,
        idxs_users=idxs_users,
        fractions=fractions,
        submodel_file_template=submodel_file_template,
        device=device,
        coalition_csv=coalition_csv,
        allocation_csv=allocation_csv,
        method="snapshot",
        max_samples=max_samples,
        save_every=save_every,
        progress=progress,
    )


__all__: Tuple[str, ...] = ("run_or", "run_mr", "run_snapshot")
