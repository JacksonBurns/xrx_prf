# xrx_prf

This repository tracks my submissions to the [OpenADMET x ExpansionRx](https://huggingface.co/spaces/openadmet/OpenADMET-ExpansionRx-Challenge) challenge.

## Usage

You need to install the following dependencies on whatever (modern) version of Python you like (I used 3.12):

```
astartes[molecules]
numpy
pandas
joblib
matplotlib
rdkit
scikit-mol
scikit-learn
xgboost
scipy
huggingface_hub
optuna
```

You will need to log in to HuggingFace (`huggingface-cli login`) on the first execution of training and inference to get permission to access the files.
After the first run, they are cached to disk.

From there you can train the models with `python training.py /path/to/output/directory` and run inference with `python inference.py /path/to/output/directory`.

## Approach

This approach uses the 'Physiochemical Random Forest' shared by BASF [here](https://github.com/JacksonBurns/chemeleon/blob/51e028a77a3cb4de87ff1e75a7ed18d4372606f4/models/rf_morgan_physchem/evaluate.py) as a starting point, swapping out a plain random forest for a more complicated hybrid model (see [`common.py`](./common.py)).

To mimic the progress of a typical drug discovery pipeline, the outputs of previous models are used as inputs for subsequent models (i.e., after training a LogD model, we use predicted LogD as an input feature to predict solubility).

## TODO

 - cache intermediate models to disk to avoid refitting them endlessly during tuning
