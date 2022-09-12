
dependencies = [ "torch", "torchvision", "yaml", "pandas"]

import yaml
from pathlib import Path
from pretrained import available_models, load_representor

BASE_DIR = Path(__file__).absolute().parents[0]

def metadata():
    with open(BASE_DIR/'metadata.yaml') as f:
        return yaml.safe_load(f)