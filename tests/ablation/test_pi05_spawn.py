"""Real Torch spawn regression; small upstream fixtures avoid loading JAX/models."""
import os
from pathlib import Path
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest

from fastwam.ablation import pi05


@pytest.mark.parametrize("name", ["compute_norm_stats", "train"])
def test_official_script_objects_survive_real_torch_dataloader_spawn(tmp_path, name):
    pytest.importorskip("torch")
    upstream = tmp_path / "upstream"
    scripts = upstream / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "__init__.py").write_text("", encoding="utf-8")
    # The statistics fixture uses the released RemoveStrings call body. The
    # training fixture exercises both a module-level class and a function; this
    # is a serialization regression, not an official model/statistics run.
    module_source = textwrap.dedent('''\
        import numpy as np

        class RemoveStrings:
            def __call__(self, x):
                return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}

        def transform(x):
            return RemoveStrings()(x)
        ''')
    (scripts / f"{name}.py").write_text(module_source, encoding="utf-8")
    harness = tmp_path / "spawn_loader.py"
    harness.write_text(textwrap.dedent('''\
        import sys
        import numpy as np
        import torch
        from fastwam.ablation.pi05 import _script

        class Dataset(torch.utils.data.Dataset):
            def __init__(self, transform):
                self.transform = transform
            def __len__(self):
                return 4
            def __getitem__(self, index):
                return self.transform({"state": np.full(8, index, np.float32),
                                       "actions": np.full((32, 8), index + 1, np.float32),
                                       "prompt": "plug transfer"})

        if __name__ == "__main__":
            module = _script(sys.argv[1], sys.argv[2])
            transform = module.RemoveStrings() if sys.argv[2] == "compute_norm_stats" else module.transform
            loader = torch.utils.data.DataLoader(Dataset(transform), batch_size=2, num_workers=2,
                                                  multiprocessing_context="spawn", timeout=45)
            batches = list(loader)
            assert len(batches) == 2
            assert all(set(batch) == {"state", "actions"} for batch in batches)
            torch.testing.assert_close(torch.cat([b["state"][:, 0] for b in batches]), torch.arange(4).float())
            torch.testing.assert_close(torch.cat([b["actions"][:, 0, 0] for b in batches]), torch.arange(1, 5).float())
            assert module.__name__ == "scripts." + sys.argv[2]
            print("SPAWN_TWO_WORKERS_PASSED")
        '''), encoding="utf-8")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(str(Path(p).resolve()) for p in sys.path if p)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run([sys.executable, "-B", str(harness), str(upstream), name],
                            cwd=tmp_path, env=environment, text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    assert "SPAWN_TWO_WORKERS_PASSED" in result.stdout


@pytest.mark.parametrize("module_name", ["scripts", "scripts.compute_norm_stats", "scripts.train"])
def test_script_loader_rejects_an_already_imported_other_source(tmp_path, monkeypatch, module_name):
    wrong = SimpleNamespace(__file__=str(tmp_path / "other" / "__init__.py"))
    monkeypatch.setitem(sys.modules, module_name, wrong)
    name = "train" if module_name == "scripts.train" else "compute_norm_stats"
    with pytest.raises(RuntimeError, match="already imported"):
        pi05._script(tmp_path / "upstream", name)


def test_script_loader_rejects_nonofficial_names(tmp_path):
    with pytest.raises(ValueError, match="Unsupported"):
        pi05._script(tmp_path, "../other")
