import json
from pathlib import Path

def save_model_mapping(labels, path: Path):
    mapping = {i: label for i, label in enumerate(sorted(set(labels)))}
    
    with open(path, "w") as f:
        json.dump(mapping, f)

    return mapping
