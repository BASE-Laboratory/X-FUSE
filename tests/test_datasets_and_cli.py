"""Unit tests for open dataset fetcher/checksum validation, utils helpers,
the `xfuse` CLI entrypoint, and Jupyter notebook hygiene.
"""

from __future__ import annotations

import ast
import hashlib
import io
import json
from pathlib import Path
from unittest.mock import patch

import h5py
import nbformat
import numpy as np
import pytest
import torch

from x_fuse.alibi import AlibiVitWrapper
from x_fuse.cli import main as cli_main
from x_fuse.datasets import (
    fetch_open_dataset,
    get_repo_root,
    get_sample_dataset_path,
    list_open_datasets,
    make_experimental_holdout,
)
from x_fuse.utils import get_alibi_model, guide_to_tensor, load_xrd_h5


def test_list_and_fetch_open_dataset_md5_verification(tmp_path: Path):
    reg = list_open_datasets()
    assert "edxia_cement_sem_eds" in reg
    assert "isis_imat_bragg_edge" in reg

    with pytest.raises(KeyError, match="Unknown dataset"):
        fetch_open_dataset("nonexistent_dataset_xyz", dest_root=tmp_path)

    with pytest.raises(FileNotFoundError, match="not found"):
        get_sample_dataset_path("NonExistentSample")

    payload_bytes = b"synthetic correlative map bytes"
    good_md5 = hashlib.md5(payload_bytes).hexdigest()
    fake_record = {
        "files": [
            {
                "key": "map_ca.tif",
                "checksum": f"md5:{good_md5}",
                "links": {"self": "https://zenodo.org/api/files/fake/map_ca.tif"},
            },
            {
                "key": "ignored.txt",
                "checksum": f"md5:{good_md5}",
                "links": {"self": "https://zenodo.org/api/files/fake/ignored.txt"},
            },
        ]
    }

    class _FakeResp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    def _fake_urlopen(url: str, timeout: int = 30):
        return _FakeResp(json.dumps(fake_record).encode("utf-8"))

    def _fake_retrieve(url: str, filename: Path):
        Path(filename).write_bytes(payload_bytes)
        return str(filename), None

    with (
        patch("urllib.request.urlopen", side_effect=_fake_urlopen),
        patch("urllib.request.urlretrieve", side_effect=_fake_retrieve) as mock_dl,
    ):
        paths = fetch_open_dataset(
            "edxia_cement_sem_eds", dest_root=tmp_path, filenames=["map_ca.tif"]
        )
        assert len(paths) == 1
        assert paths[0].read_bytes() == payload_bytes
        assert mock_dl.call_count == 1

        # Second call hits local cache without re-downloading
        paths_cached = fetch_open_dataset(
            "edxia_cement_sem_eds", dest_root=tmp_path, filenames=["map_ca.tif"]
        )
        assert paths_cached == paths
        assert mock_dl.call_count == 1

    # Verify corrupted MD5 raises RuntimeError and removes corrupt file
    bad_record = {
        "files": [
            {
                "key": "corrupt.tif",
                "checksum": "md5:00000000000000000000000000000000",
                "links": {"self": "https://zenodo.org/api/files/fake/corrupt.tif"},
            }
        ]
    }
    with (
        patch(
            "urllib.request.urlopen",
            side_effect=lambda *a, **kw: _FakeResp(
                json.dumps(bad_record).encode("utf-8")
            ),
        ),
        patch("urllib.request.urlretrieve", side_effect=_fake_retrieve),
    ):
        with pytest.raises(RuntimeError, match="MD5 mismatch"):
            fetch_open_dataset("edxia_cement_sem_eds", dest_root=tmp_path)
        assert not (tmp_path / "edxia_cement_sem_eds" / "corrupt.tif").exists()


def test_make_experimental_holdout_edge_cases():
    hr = np.random.default_rng(0).random((35, 37), dtype=np.float32)
    guide_2d = np.random.default_rng(1).random((35, 37), dtype=np.float32)

    hr_c, coarse_2d, gt_2d = make_experimental_holdout(
        hr, guide_2d, factor=4, poisson_counts=100.0, seed=7
    )
    assert hr_c.shape == (32, 36)
    assert gt_2d.shape == (32, 36)
    assert coarse_2d.shape == (8, 9)

    with pytest.raises(ValueError, match="factor must be >= 1"):
        make_experimental_holdout(hr, guide_2d, factor=0)

    with pytest.raises(ValueError, match="smaller than downsample factor"):
        make_experimental_holdout(hr[:3, :3], guide_2d[:3, :3], factor=8)

    with pytest.raises(ValueError, match="Expected 2D or 3D"):
        make_experimental_holdout(hr, np.zeros((1, 2, 16, 16)), factor=2)


def test_utils_h5_flat_guide_and_alibi_checkpoint_loader(tmp_path: Path):
    h5_file = tmp_path / "sample_xrd.h5"
    arr = np.arange(16, dtype=np.float32).reshape(4, 4)
    with h5py.File(h5_file, "w") as f:
        f.create_dataset("xrd", data=arr)
    loaded = load_xrd_h5(str(h5_file), key="xrd")
    assert np.allclose(loaded, arr)

    # Zero-variance flat guide maps cleanly to zeros without NaNs
    flat_t = guide_to_tensor(np.full((8, 8), 5.0, dtype=np.float32), normalize=True)
    assert flat_t.shape == (1, 1, 8, 8)
    assert torch.all(flat_t == 0.0)

    with pytest.raises(ValueError, match="Expected 2D, 3D, or 4D"):
        guide_to_tensor(torch.zeros(1, 1, 2, 4, 4))

    # Offline checkpoint save + get_alibi_model reload (mocking pretrained=False)
    ref_model = AlibiVitWrapper(
        "vit_small_patch14_reg4_dinov2.lvd142m",
        stride=14,
        pretrained=False,
        slope_type="constant",
        n_reg_tokens=4,
    )
    ckpt_path = tmp_path / "alibi_vits.pt"
    torch.save(ref_model.state_dict(), ckpt_path)

    orig_init = AlibiVitWrapper.__init__

    def _offline_init(self, *args, **kwargs):
        kwargs["pretrained"] = False
        return orig_init(self, *args, **kwargs)

    with patch.object(AlibiVitWrapper, "__init__", _offline_init):
        reloaded = get_alibi_model("alibi_vits", str(ckpt_path), device="cpu", stride=7)
    assert reloaded.stride == 7


def test_xfuse_cli_commands(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    assert cli_main(["info"]) == 0
    info_json = json.loads(capsys.readouterr().out)
    assert "x_fuse_version" in info_json

    assert cli_main(["datasets"]) == 0
    assert "edxia_cement_sem_eds" in capsys.readouterr().out

    assert cli_main(["datasets", "--json"]) == 0
    ds_json = json.loads(capsys.readouterr().out)
    assert "isis_imat_bragg_edge" in ds_json

    demo_out = tmp_path / "demo_summary.json"
    assert cli_main(["demo", "--epochs", "2", "--output", str(demo_out)]) == 0
    capsys.readouterr()
    summary = json.loads(demo_out.read_text(encoding="utf-8"))
    assert 0.0 <= summary["xfuse_ece"] <= 1.0


def test_tutorial_notebooks_are_valid_stripped_and_clean():
    nb_dir = get_repo_root() / "notebooks"
    notebooks = sorted(nb_dir.glob("*.ipynb"))
    assert len(notebooks) >= 2, "Expected tutorial notebooks in notebooks/"

    forbidden_tokens = (
        "/dls/science/",
        "from hr_dv2",
        "import hr_dv2",
        "import featup",
    )
    for nb_path in notebooks:
        nb = nbformat.read(nb_path, as_version=4)
        nbformat.validate(nb)
        for idx, cell in enumerate(nb.cells):
            for tok in forbidden_tokens:
                assert tok not in cell.source, (
                    f"Legacy token {tok!r} found in {nb_path.name} cell {idx}"
                )
            if cell.cell_type == "code":
                assert cell.outputs == [], (
                    f"Unstripped outputs in {nb_path.name} cell {idx}; run nbstripout"
                )
                assert cell.execution_count is None
                ast.parse(cell.source, filename=f"{nb_path.name}#cell-{idx}")
