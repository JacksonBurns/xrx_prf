# load and run inference using the trained models
import sys
from datetime import datetime

import pandas as pd
import joblib
from pathlib import Path

from common import PreviousModelTransformer, clean_smiles
from tqdm import tqdm

if __name__ == "__main__":
    try:
        model_dir = Path(sys.argv[1])
    except:
        print("Usage: python inference.py <model_directory>")
        exit(1)

    data_cache_f = Path("expansion_data_test_blinded.csv")
    if data_cache_f.exists():
        df = pd.read_csv(data_cache_f)
    else:
        df = pd.read_csv(
            "hf://datasets/openadmet/openadmet-expansionrx-challenge-test-data-blinded/expansion_data_test_blinded.csv"
        )
        df.to_csv(data_cache_f, index=False)

    test_smiles = list(map(clean_smiles, df["SMILES"]))

    out_data = {
        "Molecule Name": df["Molecule Name"],
        "SMILES": df["SMILES"],  # original smiles for output, cleaned ones for inference
    }

    targets = list(model_dir.glob("Log*"))
    pbar = tqdm(total=len(targets))
    for target in targets:
        target_name = target.stem
        pbar.set_description(f"Predicting '{target_name}'")
        pipe = joblib.load(target / "model.joblib")
        pred = pipe.predict(test_smiles)
        if "Log" in target_name and target_name != "LogD":
            pred = 10**pred
            if "Log1" in target_name:
                pred = pred - 1
            target_name = target_name.replace("Log1", "").replace("Log", "")
        out_data[target_name] = pred
        pbar.update(1)
    pbar.close()

    # timestamped result file
    out_df = pd.DataFrame(out_data)
    out_df.to_csv(
        model_dir / f"test_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", index=False
    )
