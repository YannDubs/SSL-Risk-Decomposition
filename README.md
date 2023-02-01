# Evaluating Self-Supervised Learning via Risk Decomposition [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/YannDubs/lossyless/blob/main/LICENSE) [![Python 3.8+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)

This repository contains:
- [simple API to load](#all-pretrained-models-hyperparameters-results) 169 pretrained SSL models, their pretraining hyperparameters, and all results.
- the code to compute the loss decomposition of any SSL model.
- the code to reproduce all results and figures from [Evaluating Self-Supervised Learning via Risk Decomposition](URL)

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
- `open_clip_torch`: for OpenCLIP
- `git+https://github.com/openai/CLIP.git`: for CLIP 

</details>

## Computing the loss decomposition
Here's a minimal code to compute the loss decomposition. 
```python


```

For a minimal notebook computing the loss decomposition see:

For the actual code that we used (includes hyperparameter tuning) see:

## Reproducing results


## Running
- run any script you want in the `scripts` folder with the correct server (see `config/server`). E.g. `scripts/vit.sh -s `

# Installation


NB timm is now using multi-weight support and thus changed the string names to their model. If you have issues use `timm==0.6.x`

## Contributing

If you have a pretrained model that you would like to add, please open a PR with the following:
1. In `hub/` the files and code to load your model. Then in `hubconf.py` add a one line function that loads the desired model. The name of that function will be the name of the model in `torch.hub`. Make sure that you load everything from `hub/` using a learning underscore. Follow previous examples.
2. Add all the hyperparameters and metadata in `metadata.yaml`. Documentation of every field can be found at the top of that file. Follow previous examples. 