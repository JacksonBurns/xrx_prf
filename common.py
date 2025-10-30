# adapted from here under the terms of the MIT license:
# https://github.com/JacksonBurns/chemeleon/blob/51e028a77a3cb4de87ff1e75a7ed18d4372606f4/models/rf_morgan_physchem/evaluate.py
from pathlib import Path
import sqlite3
from typing import Literal

import numpy as np
import joblib
from rdkit.Chem import MolToSmiles
from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover
from scikit_mol.conversions import SmilesToMolTransformer
from scikit_mol.descriptors import MolecularDescriptorTransformer
from scikit_mol.fingerprints import MorganFingerprintTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import QuantileTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor, XGBRFRegressor
from scipy.stats import pearsonr
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
)
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


def clean_smiles(
    smiles: str,
    remove_hs: bool = True,
    strip_stereochem: bool = False,
    strip_salts: bool = True,
) -> str:
    """Applies preprocessing to SMILES strings, seeking the 'parent' SMILES

    Note that this is different from simply _neutralizing_ the input SMILES - we attempt to get the parent molecule, analogous to a molecular skeleton.
    This is adapted in part from https://rdkit.org/docs/Cookbook.html#neutralizing-molecules

    Args:
        smiles (str): input SMILES
        remove_hs (bool, optional): Removes hydrogens. Defaults to True.
        strip_stereochem (bool, optional): Remove R/S and cis/trans stereochemistry. Defaults to False.
        strip_salts (bool, optional): Remove salt ions. Defaults to True.

    Returns:
        str: cleaned SMILES
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None, f"Could not parse SMILES {smiles}"
        if remove_hs:
            mol = Chem.RemoveHs(mol)
        if strip_stereochem:
            Chem.RemoveStereochemistry(mol)
        if strip_salts:
            remover = SaltRemover()  # use default saltremover
            mol = remover.StripMol(mol)  # strip salts

        pattern = Chem.MolFromSmarts(
            "[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]"
        )
        at_matches = mol.GetSubstructMatches(pattern)
        at_matches_list = [y[0] for y in at_matches]
        if len(at_matches_list) > 0:
            for at_idx in at_matches_list:
                atom = mol.GetAtomWithIdx(at_idx)
                chg = atom.GetFormalCharge()
                hcount = atom.GetTotalNumHs()
                atom.SetFormalCharge(0)
                atom.SetNumExplicitHs(hcount - chg)
                atom.UpdatePropertyCache()
        out_smi = Chem.MolToSmiles(
            mol, kekuleSmiles=True
        )  # this also canonicalizes the input
        assert len(out_smi) > 0, f"Could not convert molecule to SMILES {smiles}"
        return out_smi
    except Exception as e:
        print(f"Failed to clean SMILES {smiles} due to {e}")
        return None


def get_prf_pipe(
    n_estimators: int = 500,
    random_seed: int = 42,
    extra_transformers: list = [],
) -> Pipeline:
    """Returns a Physiochemical Random Forest scikit-mol pipeline, which can be fit with `pipe.fit(train_smiles, targets)`

    Args:
        n_estimators (int, optional): Number of estimators used in random forest. Defaults to 500.
        random_seed (int, optional): Random seed for reproducibility. Defaults to 42.
        extra_transformers (list, optional): List of (name, transformer) tuples to add to the feature union. Defaults to [].

    Raises:
        TypeError: Unsupported task type specified

    Returns:
        Pipeline: scikit-mol based training Pipeline for fitting with sklearn
    """
    base_models = [
        (
            "rf",
            RandomForestRegressor(
                n_estimators=n_estimators, random_state=random_seed, n_jobs=-1
            ),
        ),
        (
            "xgb",
            XGBRegressor(
                n_estimators=n_estimators, random_state=random_seed, n_jobs=-1
            ),
        ),
        (
            "xgbrf",
            XGBRFRegressor(
                n_estimators=n_estimators, random_state=random_seed, n_jobs=-1
            ),
        ),
    ]
    # Meta-regressor
    meta_model = MLPRegressor(
        hidden_layer_sizes=(8, 8, 8), random_state=random_seed, early_stopping=True
    )
    # Stacking regressor
    stacking = StackingRegressor(
        estimators=base_models,
        final_estimator=meta_model,
        passthrough=False,
        n_jobs=-1,
    )
    # Final model with target transformation
    model = TransformedTargetRegressor(
        regressor=stacking,
        transformer=QuantileTransformer(output_distribution="normal"),
    )

    pipe = Pipeline(
        [
            ("smiles2mol", SmilesToMolTransformer()),
            (
                "mol2features",
                FeatureUnion(
                    [
                        (
                            "morgan",
                            MorganFingerprintTransformer(
                                fpSize=2048,
                                radius=2,
                                useCounts=True,
                                n_jobs=-1,
                            ),
                        ),
                        (
                            "physchem",
                            MolecularDescriptorTransformer(
                                desc_list=[
                                    desc
                                    for desc in MolecularDescriptorTransformer().available_descriptors
                                    if desc
                                    != "Ipc"  # Ipc frequently has overflow issues for larger molecules
                                ],
                                n_jobs=-1,
                            ),
                        ),
                    ]
                    + extra_transformers
                ),
            ),
            # remove zero-variance features. Usually, doesn't really improve accuracy
            # but makes training faster.
            ("variance_filter", VarianceThreshold(0.0)),
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model),
        ],
        verbose=True,
    )
    return pipe


class PreviousModelTransformer:
    def __init__(self, model_paths: list[Path], cache_db: Path = Path("model_cache.sqlite")):
        self.model_paths = model_paths
        self.cache_db = cache_db
        self._ensure_schema()

    def _ensure_schema(self):
        """Create cache table if it doesn't exist."""
        with sqlite3.connect(self.cache_db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    model_path TEXT,
                    smiles TEXT,
                    prediction REAL,
                    PRIMARY KEY (model_path, smiles)
                )
            """)
            conn.commit()

    def _fetch_cached_predictions(self, conn, model_path, smiles_list):
        """Retrieve cached predictions for given model and SMILES list."""
        placeholders = ",".join("?" * len(smiles_list))
        query = f"""
            SELECT smiles, prediction
            FROM predictions
            WHERE model_path = ? AND smiles IN ({placeholders})
        """
        cur = conn.execute(query, (str(model_path), *smiles_list))
        return dict(cur.fetchall())

    def _insert_predictions(self, conn, model_path, smiles, preds):
        """Insert new predictions into the cache."""
        conn.executemany(
            "INSERT OR REPLACE INTO predictions (model_path, smiles, prediction) VALUES (?, ?, ?)",
            [(str(model_path), s, float(p)) for s, p in zip(smiles, preds)]
        )
        conn.commit()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        smis = [MolToSmiles(mol[0]) for mol in X]
        preds = []

        with sqlite3.connect(self.cache_db) as conn:
            for model_path in self.model_paths:
                # 1. Check cache
                cached = self._fetch_cached_predictions(conn, model_path, smis)
                missing = [s for s in smis if s not in cached]

                # 2. Compute missing predictions
                if missing:
                    model = joblib.load(model_path)
                    new_preds = model.predict(missing).flatten()
                    self._insert_predictions(conn, model_path, missing, new_preds)
                    del model
                else:
                    new_preds = []

                # 3. Combine cached + new results
                all_preds = np.array([cached.get(s) for s in smis], dtype=np.float64)
                # Fill missing entries
                for i, s in enumerate(smis):
                    if np.isnan(all_preds[i]):
                        # find corresponding prediction from new_preds
                        idx = missing.index(s)
                        all_preds[i] = new_preds[idx]
                preds.append(all_preds)

        return np.stack(preds, axis=1)



def parity_plot(
    truth: np.ndarray,
    prediction: np.ndarray,
    title: str = "",
    quantity: str = "",
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    style: Literal["hexbin", "scatter"] = "scatter",
) -> None:
    """Create a scatter parity plot with an inset pie chart."""
    if xlim is None:
        xlim = (min(truth.min(), prediction.min()), max(truth.max(), prediction.max()))
    if ylim is None:
        ylim = xlim

    x_label = "True"
    y_label = "Predicted"
    if quantity:
        x_label += f" {quantity}"
        y_label += f" {quantity}"

    fig, ax = plt.subplots(figsize=(6, 5))
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    ax: Axes  # type hint

    if style == "hexbin":
        hb = ax.hexbin(
            truth,
            prediction,
            gridsize=80,
            cmap="viridis",
            mincnt=1,
        )
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label("Number of compounds")
    elif style == "scatter":
        ax.scatter(
            truth,
            prediction,
            s=10,
            alpha=0.15 if truth.shape[0] > 1_000 else 0.5,
            color="C0",  # Default Matplotlib blue
        )
    else:
        raise ValueError(f"Unknown style: {style}")

    mae = round(mean_absolute_error(truth, prediction), 1)

    # 1:1 line
    ax.plot(xlim, xlim, "r", linewidth=1)
    # ±mae lines
    ax.plot(xlim, (np.array(xlim) + mae), "r--", linewidth=0.5)
    ax.plot(xlim, (np.array(xlim) - mae), "r--", linewidth=0.5)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(True, which="major", axis="both")
    ax.set_axisbelow(True)

    # Text box with R² and MSE
    r = pearsonr(truth, prediction)[0]
    textstr = (
        f"$\\bf{{R^2}}:$ {r**2:.3f}\n"
        f"$\\bf{{r}}:$ {r:.3f}\n"
        f"$\\bf{{MAE}}:$ {mae:.2f}\n"
        f"$\\bf{{MSE}}:$ {mean_squared_error(truth, prediction):.2f}\n"
        f"$\\bf{{RMSE}}:$ {root_mean_squared_error(truth, prediction):.2f}\n"
        f"$\\bf{{Support}}:$ {truth.shape[0]:d}"
    )
    ax.text(
        0.05,
        0.95,
        textstr,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Inset pie chart: fraction within ±mae
    frac_within_mae = np.mean(np.abs(truth - prediction) < mae)
    sizes = [1 - frac_within_mae, frac_within_mae]
    ax_inset = ax.inset_axes([0.75, 0.025, 0.25, 0.25], transform=ax.transAxes)
    ax_inset.pie(
        sizes,
        colors=["#ae2b27", "#4073b2"],
        startangle=360 * (frac_within_mae - 0.5) / 2,
        wedgeprops={"edgecolor": "black"},
        autopct="%1.f%%",
        textprops=dict(color="w"),
    )
    ax_inset.axis("equal")
    ax_inset.set_title(f"$\\bf{{±{mae:.1f}}}$ {quantity}", fontsize=10)

    plt.tight_layout()
    return fig
