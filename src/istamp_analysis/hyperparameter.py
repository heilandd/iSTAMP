import copy
import json
import math
import os
import random
from typing import Dict, Any, Tuple

import numpy as np
import optuna
import torch

# ------------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------------
# Optional: robust metric extraction from history
# Adjust this if your history uses different keys
# ------------------------------------------------------------------
def extract_best_metric_from_history(history, metric_priority=None, mode="max"):
    """
    Tries to extract the best validation metric from the returned history object.
    Assumes history is either:
      - dict of metric_name -> list
      - list of dicts per epoch

    Returns:
      best_score: float
      best_epoch: int
      metric_name: str
    """
    if metric_priority is None:
        metric_priority = [
            "val_f1_weighted",
            "val_f1_macro",
            "val_balanced_accuracy",
            "val_accuracy",
            "val_loss",
        ]

    if history is None:
        raise ValueError("History is None; return_history=True is required.")

    # Case 1: history is dict of lists
    if isinstance(history, dict):
        for metric_name in metric_priority:
            if metric_name in history and len(history[metric_name]) > 0:
                values = history[metric_name]
                if "loss" in metric_name:
                    best_epoch = int(np.argmin(values))
                    best_score = float(values[best_epoch])
                    return best_score, best_epoch, metric_name
                else:
                    best_epoch = int(np.argmax(values))
                    best_score = float(values[best_epoch])
                    return best_score, best_epoch, metric_name

    # Case 2: history is list of dicts
    if isinstance(history, list) and len(history) > 0 and isinstance(history[0], dict):
        for metric_name in metric_priority:
            if metric_name in history[0]:
                values = [epoch_dict[metric_name] for epoch_dict in history if metric_name in epoch_dict]
                if len(values) == 0:
                    continue
                if "loss" in metric_name:
                    best_epoch = int(np.argmin(values))
                    best_score = float(values[best_epoch])
                    return best_score, best_epoch, metric_name
                else:
                    best_epoch = int(np.argmax(values))
                    best_score = float(values[best_epoch])
                    return best_score, best_epoch, metric_name

    raise ValueError(
        "Could not extract validation metric from history. "
        "Please inspect history keys and adapt extract_best_metric_from_history()."
    )


# ------------------------------------------------------------------
# Main tuning function
# ------------------------------------------------------------------
def bayesian_hyperparameter_tuning(
    dataset_hop3,
    train_loader,
    val_loader,
    bias_classes: Dict[str, int],
    train_fn,
    model_class,
    device=None,
    n_trials: int = 40,
    epochs: int = 100,
    patience: int = 15,
    study_name: str = "raman_gatv2_multibias_tuning",
    storage: str = None,
    seed: int = 42,
    lambda_bias_fixed: Dict[str, float] = None,
    optimize_metric: str = "val_f1_weighted",
    direction: str = "maximize",
):
    """
    Bayesian-style hyperparameter tuning using Optuna (TPE sampler).

    Parameters
    ----------
    dataset_hop3 : dataset
        Your graph dataset.
    train_loader, val_loader : DataLoader
        Training and validation loaders.
    bias_classes : dict
        Dict like {"Center": n, "sample_type": n, ...}
    train_fn : callable
        Your training function, e.g. train_raman_gnn_multibias
    model_class : class
        Your model class, e.g. RamanGATv2MultiBias
    device : torch.device
    n_trials : int
    epochs : int
    patience : int
    study_name : str
    storage : str or None
        Example: "sqlite:///raman_optuna.db"
    seed : int
    lambda_bias_fixed : dict or None
        If None, lambda_bias will also be tuned.
    optimize_metric : str
        One of the keys that appear in history, e.g. "val_f1_weighted"
    direction : str
        "maximize" for F1/accuracy, "minimize" for loss.
    """

    set_seed(seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    in_channels = dataset_hop3[0].x.shape[1]
    out_channels = int(dataset_hop3[0].y.numel())
    edge_dim = dataset_hop3[0].edge_attr.shape[1] if dataset_hop3[0].edge_attr is not None else 1

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)

    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
    )

    def objective(trial: optuna.Trial):
        set_seed(seed + trial.number)

        # -----------------------------
        # Hyperparameter search space
        # -----------------------------
        #hidden_channels = trial.suggest_int("hidden_channels", 64, 256, step=32)
        #num_heads = trial.suggest_int("num_heads", 2, 8)
        
        num_heads = trial.suggest_categorical("num_heads", [3, 4, 5, 6, 8])
        hidden_multiplier = trial.suggest_categorical("hidden_multiplier", [16, 20, 24, 27, 32])
        hidden_channels = num_heads * hidden_multiplier
        
        
        num_layers = trial.suggest_int("num_layers", 2, 5)
        dropout = trial.suggest_float("dropout", 0.1, 0.6)
        use_batch_norm = trial.suggest_categorical("use_batch_norm", [True, False])

        bias_hidden_dim = trial.suggest_categorical("bias_hidden_dim", [64, 128, 256])
        grl_alpha = trial.suggest_float("grl_alpha", 0.1, 2.0, log=True)

        lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True)

        # Optional tuning of lambda_bias
        if lambda_bias_fixed is None:
            lambda_bias = {
                "Center": trial.suggest_float("lambda_bias_center", 0.01, 0.5, log=True),
                "sample_type": trial.suggest_float("lambda_bias_sample_type", 0.01, 0.5, log=True),
                "image_id": trial.suggest_float("lambda_bias_image_id", 0.01, 0.5, log=True),
                "patient_id": trial.suggest_float("lambda_bias_patient_id", 0.005, 0.2, log=True),
            }
        else:
            lambda_bias = copy.deepcopy(lambda_bias_fixed)

        model = model_class(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            edge_dim=edge_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            use_batch_norm=use_batch_norm,
            bias_classes=bias_classes,
            bias_hidden_dim=bias_hidden_dim,
            grl_alpha=grl_alpha,
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        # -----------------------------
        # Train
        # -----------------------------
        model, history = train_fn(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=device,
            epochs=epochs,
            patience=patience,
            lambda_bias=lambda_bias,
            use_bias_adaptation=True,
            return_history=True,
        )

        # -----------------------------
        # Extract score
        # -----------------------------
        if optimize_metric in ["val_loss", "loss", "best_val_loss"]:
            metric_priority = ["val_loss"]
            best_score, best_epoch, metric_name = extract_best_metric_from_history(
                history, metric_priority=metric_priority, mode="min"
            )
            score = best_score
        else:
            metric_priority = [
                optimize_metric,
                "val_f1_weighted",
                "val_f1_macro",
                "val_balanced_accuracy",
                "val_accuracy",
            ]
            best_score, best_epoch, metric_name = extract_best_metric_from_history(
                history, metric_priority=metric_priority, mode="max"
            )
            score = best_score

        # Store useful trial metadata
        trial.set_user_attr("best_epoch", int(best_epoch))
        trial.set_user_attr("metric_name", metric_name)
        trial.set_user_attr("lambda_bias", lambda_bias)

        return score

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # -----------------------------
    # Print best result
    # -----------------------------
    print("\nBest trial:")
    print(f"  Value: {study.best_trial.value:.6f}")
    print("  Params:")
    for k, v in study.best_trial.params.items():
        print(f"    {k}: {v}")

    return study
