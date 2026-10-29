"""The README's minimal example must run."""

import re
import runpy
import warnings

import pytest

from cases import REPO_ROOT


@pytest.mark.slow
def test_readme_minimal_example(tmp_path, monkeypatch):
    code = re.search(r"### Minimal example\n\n```python\n(.*?)```", (REPO_ROOT / "README.md").read_text(), re.S)
    script = tmp_path / "example.py"
    script.write_text(code.group(1))
    monkeypatch.chdir(REPO_ROOT)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        namespace = runpy.run_path(str(script))
    assert namespace["result"].baseflow[:3].tolist() == [27816.77197, 25989.5, 25468.0]
    assert 0 < namespace["calibrated"].baseflow_fraction < 1
