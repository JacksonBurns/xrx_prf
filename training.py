import sys
from datetime import datetime
from pathlib import Path

from astartes.molecules import train_test_split
import numpy as np
import pandas as pd
import joblib
import matplotlib

matplotlib.use("Agg")

from common import get_prf_pipe, PreviousModelTransformer, parity_plot, clean_smiles


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
        _df = pd.read_csv(
            "hf://datasets/openadmet/openadmet-expansionrx-challenge-train-data/expansion_data_train_raw.csv"
        )
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

        # split and fit
        *_, train_idxs, val_idxs = train_test_split(
            df[SMILES_COL].to_numpy(),
            train_size=0.8,
            test_size=0.2,
            random_state=42,
            sampler="random",  # can change this to possibly improve performance
            return_indices=True,
        )
        if previous_model_paths:
            extra_transformers = [
                ("previous_models", PreviousModelTransformer(previous_model_paths, outdir / "cache.db"))
            ]
        else:
            extra_transformers = []
        pipe = get_prf_pipe(extra_transformers=extra_transformers)
        pipe.fit(df[SMILES_COL].iloc[train_idxs], df[target].iloc[train_idxs])
        val_pred = pipe.predict(df[SMILES_COL].iloc[val_idxs])

        # saving of predictions and model
        data = {"smiles": df[SMILES_COL].iloc[val_idxs].reset_index(drop=True)}
        data[f"true_{target}"] = df[target].iloc[val_idxs].reset_index(drop=True)
        data[f"pred_{target}"] = val_pred
        pd.DataFrame(data).to_csv(Path(subdir) / "val_predictions.csv", index=False)
        out_model = subdir / "model.joblib"
        joblib.dump(pipe, out_model)
        previous_model_paths.append(out_model)

        val_df = pd.read_csv(subdir / "val_predictions.csv")
        fig = parity_plot(
            val_df[f"true_{target}"],
            val_df[f"pred_{target}"],
            quantity=target,
        )
        fig.savefig(subdir / "validation_parity.png", dpi=300)
