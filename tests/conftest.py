import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS_DIR.parent))
sys.path.insert(0, str(TESTS_DIR))


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: full calibration runs (~1 min total); deselect with -m 'not slow'")
