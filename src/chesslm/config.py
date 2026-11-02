"""Load YAML configs from the repo's configs/ folder."""

from pathlib import Path

import yaml

# src/chesslm/config.py -> repo root is two folders above src/.
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


def load_config(name):
    """Return configs/<name>.yaml as a dict, e.g. load_config("eval")."""
    with open(CONFIG_DIR / f"{name}.yaml") as f:
        return yaml.safe_load(f)
