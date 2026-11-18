"""Run a JSON argument list with the active OpenPI updater trainer (torchrun compatible)."""
import json
from pathlib import Path
import sys


def main():
    if len(sys.argv) < 2:
        raise SystemExit('Usage: train_updater.py RECIPE.json [additional trainer arguments]')
    recipe = json.loads(Path(sys.argv[1]).read_text())
    if not isinstance(recipe, list) or not all(isinstance(arg, str) for arg in recipe):
        raise ValueError('The training recipe must be a JSON list of command-line arguments')
    sys.argv[1:] = recipe + sys.argv[2:]
    from openpi.dual.train import main as train
    train()


if __name__ == '__main__':
    main()
