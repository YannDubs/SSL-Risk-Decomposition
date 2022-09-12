
dependencies = [ "torch", "torchvision", "pyyaml" ]

import yaml
from pretrained import available_models, load_representor

def get_metadata():
    with open('metadata.yaml') as f:
        return yaml.safe_load(f)