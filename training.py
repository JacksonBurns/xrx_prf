import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import joblib
import matplotlib
import optuna
from sklearn.metrics import mean_absolute_error

matplotlib.use("Agg")

from common import (
    get_prf_pipe,
    PreviousModelTransformer,
    parity_plot,
    clean_smiles,
)

# from previous hpopt runs
KNOWN_PARAMS = {}

# for my initial interpolation efforts, these were the best settings
{
    "LogD":                         {'morgan_radius': 3, 'stack_chemprop': True, 'stack_xgb': True, 'stack_knn': True, 'stack_elasticnet': True, 'stack_svr': True, 'final_estimator': 'hgb', 'global_target_scaling': False},
    "KSOL":                         {'morgan_radius': 4, 'stack_chemprop': True, 'stack_xgb': True, 'stack_knn': True, 'stack_elasticnet': True, 'stack_svr': False, 'final_estimator': 'hgb', 'global_target_scaling': False},
    "MLM CLint":                    {'morgan_radius': 4, 'stack_chemprop': True, 'stack_xgb': False, 'stack_knn': False, 'stack_elasticnet': False, 'stack_svr': True, 'final_estimator': 'hgb', 'global_target_scaling': False},
    "HLM CLint":                    {'morgan_radius': 4, 'stack_chemprop': True, 'stack_xgb': True, 'stack_knn': True, 'stack_elasticnet': True, 'stack_svr': True, 'final_estimator': 'rf', 'global_target_scaling': False},
    "Caco-2 Permeability Papp A>B": {'morgan_radius': 2, 'stack_chemprop': True, 'stack_xgb': True, 'stack_knn': False, 'stack_elasticnet': True, 'stack_svr': False, 'final_estimator': 'hgb', 'global_target_scaling': False},
    "Caco-2 Permeability Efflux":   {'morgan_radius': 2, 'stack_chemprop': True, 'stack_xgb': False, 'stack_knn': True, 'stack_elasticnet': False, 'stack_svr': False, 'final_estimator': 'hgb', 'global_target_scaling': True},
    "MPPB":                         {'morgan_radius': 4, 'stack_chemprop': True, 'stack_xgb': False, 'stack_knn': False, 'stack_elasticnet': False, 'stack_svr': True, 'final_estimator': 'hgb', 'global_target_scaling': True},
    "MGMB":                         {'morgan_radius': 4, 'stack_chemprop': False, 'stack_xgb': False, 'stack_knn': True, 'stack_elasticnet': False, 'stack_svr': False, 'final_estimator': 'hgb', 'global_target_scaling': False},
    "MBPB":                         {'morgan_radius': 2, 'stack_chemprop': True, 'stack_xgb': False, 'stack_knn': True, 'stack_elasticnet': False, 'stack_svr': True, 'final_estimator': 'rf', 'global_target_scaling': False},
}


# these are in a specific order of which will be used to predict the others
TARGETS = [
    "LogD",
    "KSOL",
    "MLM CLint",
    "HLM CLint",
    "Caco-2 Permeability Papp A>B",
    "Caco-2 Permeability Efflux",
    "MPPB",
    "MGMB",
    "MBPB",
]
SMILES_COL = "SMILES"

TUNING_TRIALS = 64  # number of optuna trials for hyperparameter tuning


def define_by_run(trial):
    params = dict(
        morgan_radius=trial.suggest_categorical("morgan_radius", [2, 3, 4]),
        morgan_size=trial.suggest_categorical("morgan_size", [1024, 2048, 4096]),
        stack_chemprop=trial.suggest_categorical("stack_chemprop", [True, False]),
        stack_xgb=trial.suggest_categorical("stack_xgb", [True, False]),
        stack_knn=trial.suggest_categorical("stack_knn", [True, False]),
        stack_elasticnet=trial.suggest_categorical("stack_elasticnet", [True, False]),
        stack_svr=trial.suggest_categorical("stack_svr", [True, False]),
        final_estimator=trial.suggest_categorical("final_estimator", ["elasticnet", "hgb", "rf"]),
        global_target_scaling=trial.suggest_categorical("global_target_scaling", [True, False]),
    )

    if params["final_estimator"] == "hgb":
        params["hgb_max_depth"] = trial.suggest_int("hgb_max_depth", 3, 10)
        params["hgb_learning_rate"] = trial.suggest_categorical("hgb_learning_rate", [0.001, 0.01, 0.05, 0.1])
    else:
        params["hgb_max_depth"] = 3
        params["hgb_learning_rate"] = 0.05

    # Conditionally suggest hyperparameters for stacked models
    if params["stack_xgb"]:
        params["xgb_max_depth"] = trial.suggest_int("xgb_max_depth", 3, 10)
        params["xgb_learning_rate"] = trial.suggest_float("xgb_learning_rate", 1e-3, 0.3, log=True)
    else:
        # Provide a default value if not stacked, though common.py uses its own defaults
        # This is primarily to avoid Optuna errors if the parameter isn't defined.
        params["xgb_max_depth"] = 6
        params["xgb_learning_rate"] = 0.1

    if params["stack_knn"]:
        params["knn_n_neighbors"] = trial.suggest_int("knn_n_neighbors", 3, 30)
    else:
        params["knn_n_neighbors"] = 8

    if params["stack_svr"]:
        params["svr_C"] = trial.suggest_float("svr_C", 1e-1, 100.0, log=True)
        params["svr_gamma"] = trial.suggest_categorical("svr_gamma", ["scale", "auto"])
    else:
        params["svr_C"] = 1.0
        params["svr_gamma"] = "scale"
    
    if params["stack_chemprop"]:
        params["chemeleon_ffn_hidden_dim"] = trial.suggest_int("chemeleon_ffn_hidden_dim", 400, 3000, 200)
        params["chemeleon_ffn_num_layers"] = trial.suggest_int("chemeleon_ffn_num_layers", 0, 3)

    return params


def train_one(
    df,
    train_idxs,
    val_idxs,
    target,
    subdir,
    extra_transformers,
    write_output=False,
    **kwargs,
):
    pipe = get_prf_pipe(
        extra_transformers=extra_transformers,
        **kwargs,
    )
    pipe.fit(df[SMILES_COL].iloc[train_idxs], df[target].iloc[train_idxs])
    val_pred = pipe.predict(df[SMILES_COL].iloc[val_idxs])
    data = {"smiles": df[SMILES_COL].iloc[val_idxs].reset_index(drop=True)}
    data[f"true_{target}"] = df[target].iloc[val_idxs].reset_index(drop=True)
    data[f"pred_{target}"] = val_pred
    val_df = pd.DataFrame(data)
    if write_output:
        val_df.to_csv(Path(subdir) / "val_predictions.csv", index=False)
        joblib.dump(pipe, subdir / "validation_model.joblib")
        fig = parity_plot(
            val_df[f"true_{target}"],
            val_df[f"pred_{target}"],
            quantity=target,
        )
        fig.savefig(subdir / "validation_parity.png", dpi=300)
    return mean_absolute_error(val_df[f"true_{target}"], val_df[f"pred_{target}"])


if __name__ == "__main__":
    try:
        outdir = Path(sys.argv[1])
    except:
        print("Usage: python training.py <output_directory>")
        exit(1)

    # timestamped output directory
    outdir /= datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    outdir.mkdir()

    # get the data
    data_cache_f = Path("expansion_data_train_raw.csv")
    if data_cache_f.exists():
        _df = pd.read_csv(data_cache_f)
    else:
        _df = pd.read_csv("hf://datasets/openadmet/openadmet-expansionrx-challenge-train-data/expansion_data_train_raw.csv")
        _df.to_csv(data_cache_f, index=False)

    _df[SMILES_COL] = _df[SMILES_COL].map(clean_smiles)
    _df = _df[_df[SMILES_COL].map(lambda x: x is not None)]

    # going to fit one model per target, re-using previous models outputs on subsequent models
    previous_model_paths = []
    for _target in TARGETS:
        df = _df.copy()

        # log (possibly +1) transform those which are not already logged
        if "Log" not in _target:
            if (df[_target] == 0.0).any():
                target = "Log1" + _target
                df[target] = np.log10(1 + df[_target])
            else:
                target = "Log" + _target
                df[target] = np.log10(df[_target])
        else:
            target = _target

        # just in case
        df[target] = df[target].replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=[target])

        subdir = outdir / target.replace(" ", "_")
        subdir.mkdir(parents=True, exist_ok=True)

        extra_transformers = []
        if previous_model_paths:
            extra_transformers += [
                (
                    "previous_models",
                    PreviousModelTransformer(previous_model_paths, outdir / "cache.db"),
                )
            ]

        # start by hyperparameter optimizing the model
        # perform roughly time-based splitting (roughly, because it's based on the molecule label)
        df = df.sort_values("Molecule Name")  # already sorted, but just in case
        _cutoff = int(df.shape[0] * 0.80)
        train_idxs = np.arange(_cutoff)
        val_idxs = np.arange(_cutoff, df.shape[0])

        if _target in KNOWN_PARAMS:
            # mock the outcome of the study with known params
            study = SimpleNamespace()
            study.best_params = KNOWN_PARAMS[_target]
        else:
            study = optuna.create_study(direction="minimize")
            study.optimize(
                lambda trial: train_one(
                    df,
                    train_idxs,
                    val_idxs,
                    target,
                    subdir,
                    extra_transformers,
                    write_output=False,
                    **define_by_run(trial),
                ),
                n_trials=TUNING_TRIALS,
            )
            with open(subdir / f"optuna_study_{target.replace(' ', '_')}.txt", "w") as f:
                f.write(f"Best hyperparameters for target {target}: {study.best_params}\n")
            study.trials_dataframe().to_csv(subdir / f"optuna_study_results_{target.replace(' ', '_')}.csv")

        # for reference, train and save the validation model with the optimal settings
        train_one(
            df,
            train_idxs,
            val_idxs,
            target,
            subdir,
            extra_transformers,
            write_output=True,
            **study.best_params,
        )

        # using the optimal settings, train a model on the entire dataset for actual submission
        pipe = get_prf_pipe(extra_transformers=extra_transformers, random_seed=42, **study.best_params)
        pipe.fit(df[SMILES_COL], df[target])
        outmodel = subdir / "final_model.joblib"
        joblib.dump(pipe, outmodel)
        previous_model_paths.append(outmodel)
