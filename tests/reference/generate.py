"""Regenerate the characterization references in tests/reference/.

Run from the repo root:

    PYTHONPATH=. python tests/reference/generate.py            # all cases
    PYTHONPATH=. python tests/reference/generate.py kernels    # selected cases
    PYTHONPATH=. python tests/reference/generate.py --check-determinism

Only regenerate on purpose: the references pin baseline-original behavior, and
overwriting them after a refactor would hide any change the refactor made.
"""

import argparse
import contextlib
import io
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root (pybfs)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # tests/ (cases)
import cases  # noqa: E402

REFERENCE_DIR = Path(__file__).resolve().parent


def versions():
    import numba
    import pandas
    import scipy
    import statsmodels
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pandas.__version__,
        "scipy": scipy.__version__,
        "numba": numba.__version__,
        "statsmodels": statsmodels.__version__,
        "platform": platform.platform(),
    }


def run_case(name):
    """Run a case with its stdout (model diagnostics) suppressed."""
    with contextlib.redirect_stdout(io.StringIO()):
        start = time.perf_counter()
        result = cases.ALL_CASES[name]()
        elapsed = time.perf_counter() - start
    return {k: np.asarray(v) for k, v in result.items()}, elapsed


def identical(a, b):
    if a.keys() != b.keys():
        return False
    return all(np.array_equal(a[k], b[k], equal_nan=a[k].dtype.kind == "f") for k in a)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*", help="cases to regenerate (default: all)")
    parser.add_argument("--check-determinism", action="store_true",
                        help="run each case twice and confirm the outputs are bit-identical; writes nothing")
    args = parser.parse_args()
    names = args.names or list(cases.ALL_CASES)

    meta_path = REFERENCE_DIR / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"cases": {}}
    for name in names:
        result, elapsed = run_case(name)
        if args.check_determinism:
            again, _ = run_case(name)
            print(f"{name:24s} {elapsed:7.2f} s  bit-identical on rerun: {identical(result, again)}")
            continue
        np.savez_compressed(REFERENCE_DIR / f"{name}.npz", **result)
        meta["cases"][name] = {"seconds": round(elapsed, 2), "arrays": len(result)}
        print(f"{name:24s} {elapsed:7.2f} s  {len(result)} arrays")
    if not args.check_determinism:
        meta["versions"] = versions()
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
