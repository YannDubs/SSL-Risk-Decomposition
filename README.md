# Evaluating Self-Supervised Learning via Risk Decomposition [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/YannDubs/lossyless/blob/main/LICENSE) [![Python 3.8+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)

This repository contains:
- [simple API to load](#all-pretrained-models-hyperparameters-results) 169 pretrained SSL models, their pretraining hyperparameters, and all results.
- [simple code](#computing-the-loss-decomposition) to compute the loss decomposition of any SSL model.
- [the code](#reproducing-results) to reproduce all results and figures from [Evaluating Self-Supervised Learning via Risk Decomposition](URL)

## All pretrained models, hyperparameters, results

We release all pretrained weights, hyperparameters, and results on `torch.hub`, which can be loaded using:

```python
import torch

# loads the desired pretrained model and preprocessing pipeline
name = "dino_rn50" # example
model, preprocessor = torch.hub.load('YannDubs/SSL-Risk-Decomposition:main', name, trust_repo=True)

# gets all available models 
available_names = torch.hub.list('YannDubs/SSL-Risk-Decomposition:main')

# gets all results and hyperparameters as a dataframe 
results_df = torch.hub.load('YannDubs/SSL-Risk-Decomposition:main', "results_df")
```

The necessary dependencies are: 
- for **most models**: `pip install torch torchvision tqdm timm pandas`
- for **all models**: `pip install torch torchvision tqdm timm dill open_clip_torch git+https://github.com/openai/CLIP.git`
<details>
  <summary><b>Details</b></summary>
    
- `timm`: for any ViT architecture
- `pandas`: for results_df, metadata_df
- `dill`: for BYOL
- `open-clip-torch`: for OpenCLIP
- `git+https://github.com/openai/CLIP.git`: for CLIP 

</details>

## Computing the loss decomposition
Here's a minimal code to compute the loss decomposition. 
```python

def compute_risk_components(model_ssl, D_train, D_test, model_sup=None, n_sub=10000, **kwargs):
    """Computes the SSL risk decomposition for `model_ssl` using a given training and testing set.
    
    If we are given a supervised `model_sup` of the same architecture as model_ssl, we compute the 
    approximation error. Else we merge it with usability error given that approx error is neglectable.
    """
    errors = dict()
    
    # featurize data to make probing much faster. Optional.
    D_train = featurize_data(model_ssl, D_train)
    D_test = featurize_data(model_ssl, D_test)
    
    D_comp, D_sub = data_split(D_train, n=n_sub)
    
    r_A_F = train_eval_probe(D_train, D_train, **kwargs)
    r_A_S = train_eval_probe(D_comp, D_sub, **kwargs)
    r_U_S = train_eval_probe(D_train, D_test, **kwargs)
    
    if model_sup is not None:
        D_train_sup = featurize_data(model_sup, D_train)
        errors["approx"] = train_eval_probe(D_train_sup, D_train_sup, **kwargs)
        errors["usability"] = r_A_F - errors["approx"]
    else:
        errors["usability"] = r_A_F # merges both errors but approx is neglectable
        
    errors["probe_gen"] = r_A_S - r_A_F
    errors["encoder_gen"] = r_U_S - r_A_S 
    errors["agg_risk"] = r_U_S
    return errors

def featurize_data(model, dataset):
    """Featurize a dataset using the model."""
    ...


def train_eval_probe(D_train, D_test, **kwargs):
    """Trains a model (encoder and probe) on D_train and evaluates it on D_test"""
    ...

def data_split(dataset, n):
    """Split a dataset into a set of size n and its complement"""
    ...
```

For a minimal notebook computing the loss decomposition and a specific implementations for the above functions see: [![Minimal training of DISSL](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YannDubs/SSL-Risk-Decomposition/blob/main/notebooks/minimal.ipynb).

For the actual code that we used (includes hyperparameter tuning) see: [main_fullshot.py](https://github.com/YannDubs/SSL-Risk-Decomposition/blob/main/main_fullshot.py)

## Reproducing results

Steps to reproduce all the paper:
0. Install `requirements_running.txt` (pip) or `environment_running.yml` (conda) to compute all risk components.
1. to recompute all risk components run: `scripts/run_all.sh`. To recompute specific models the corresponding script in `scripts/` with the correct server (see `config/server`). E.g. `scripts/simsiam.sh -s nlprun`
2. to recompute all few shot evaluation run: `script_sk/run_all.sh` (we use sklearn instead of pytorch for that).
3. Install `requirements_analyzing.txt` (pip) or `environment_analyzing.yml` (conda) to analyze all results.
4. to reproduce all the analysis and plot from the main_paper run `notebooks/main_paper.ipynb`
5. to reproduce all the analysis and plot from the appendices run `notebooks/appcs.ipynb`

## Contributing

If you have a pretrained model that you would like to add, please open a PR with the following:
1. In `hub/` the files and code to load your model. Then in `hubconf.py` add a one line function that loads the desired model. The name of that function will be the name of the model in `torch.hub`. Make sure that you load everything from `hub/` using a learning underscore. Follow previous examples.
2. Add all the hyperparameters and metadata in `metadata.yaml`. Documentation of every field can be found at the top of that file. Follow previous examples.