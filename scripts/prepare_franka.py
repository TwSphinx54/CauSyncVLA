"""Materialize the FR3 OpenPI profile in a separate Python import directory."""
import argparse
from pathlib import Path
import shutil
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path, nargs='?', default=Path('.runtime/franka'))
    destination = parser.parse_args().destination.resolve()
    if destination.exists():
        parser.error(f'{destination} exists; use a fresh output directory')
    root = Path(__file__).resolve().parents[1]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as work:
        stage = Path(work) / 'profile'
        ignore = shutil.ignore_patterns('__pycache__', '*.pyc')
        shutil.copytree(root / 'src/openpi', stage / 'openpi', ignore=ignore)
        shutil.copytree(root / 'franka/overlay/openpi', stage / 'openpi', ignore=ignore, dirs_exist_ok=True)
        stage.rename(destination)
    print(destination)


if __name__ == '__main__':
    main()
