"""Portable locations for generated datasets, representations and model outputs."""
import os
from pathlib import Path
REPOSITORY = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get('PROTEIN_INPUT_WORKDIR', str(REPOSITORY / 'work'))).expanduser().resolve()
TASKS = ['EC_level2', 'GO_slim', 'Subcellular']
