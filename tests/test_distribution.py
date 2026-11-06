"""What a clone has to contain for someone else to reproduce this work."""
from __future__ import annotations

import csv
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_MANIFEST = ROOT / "weights" / "MANIFEST.tsv"
PATCH = ROOT / "external" / "structtokenbench.patch"


def _tracked() -> set[str]:
    result = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files"], capture_output=True, text=True
    )
    if result.returncode != 0:
        pytest.skip("not a git checkout")
    return set(result.stdout.split())


def test_the_weights_manifest_is_tracked_even_though_the_weights_are_not():
    """A clone must be able to say what it is missing, and check what it receives."""
    tracked = _tracked()
    assert "weights/MANIFEST.tsv" in tracked
    assert not [p for p in tracked if p.endswith(".pt")], "weights must not be committed"


def test_every_released_file_has_a_size_and_a_hash():
    with WEIGHTS_MANIFEST.open() as handle:
        rows = list(csv.DictReader(
            (line for line in handle if not line.startswith("#")), delimiter="\t"))
    assert rows, "the weights manifest lists nothing"
    for row in rows:
        assert row["path"] and not row["path"].startswith("/")
        assert int(row["bytes"]) > 0
        assert re.fullmatch(r"[0-9a-f]{64}", row["sha256"]), row["path"]


def test_documented_weights_match_the_manifest():
    """Documented weights must be available through the manifest."""
    readme = (ROOT / "README.md").read_text()
    released = set(re.findall(r"`(weights/[\w/]+\.pt)`", readme))
    with WEIGHTS_MANIFEST.open() as handle:
        listed = {
            "weights/" + row["path"]
            for row in csv.DictReader(
                (line for line in handle if not line.startswith("#")), delimiter="\t")
            if row.get("path")
        }
    assert released, "the README names no released weights file"
    assert released <= listed, f"released but not in the manifest: {sorted(released - listed)}"


def test_the_benchmark_patch_is_shipped_and_names_its_base():
    """`git clone` of the benchmark is not enough, so the patch travels with this repository."""
    assert PATCH in {ROOT / p for p in map(Path, _tracked())}, "the patch is not tracked"
    header = PATCH.read_text()[:4000]
    assert re.search(r"^Base commit: [0-9a-f]{40}$", header, flags=re.M), \
        "the patch must record the upstream commit it was generated against"
    assert "MIT License" in header, "the patch must carry the upstream attribution"
    assert "diff --git" in PATCH.read_text(), "the patch has no diff body"


def test_the_patch_covers_what_the_integration_check_requires():
    """The two must not drift: a patch that no longer satisfies the check is worse than none."""
    body = PATCH.read_text()
    from importlib.machinery import SourceFileLoader

    module = SourceFileLoader(
        "check_integration", str(ROOT / "scripts/structtokenbench/check_integration.py")
    ).load_module()
    for relative, requirements in module.REQUIRED.items():
        assert relative in body, f"the patch does not touch {relative}"
        for needle, _why in requirements:
            assert f"+{needle}" in body or needle in body, \
                f"the patch does not add `{needle}`, which check_integration.py requires"


def test_both_dataset_builders_are_present():
    """The decoder's corpus and the encoder's corpus need different builders."""
    assert (ROOT / "scripts" / "build_cache.py").is_file(), "no CATH cache builder"
    assert (ROOT / "scripts" / "build_afdb_cache.py").is_file(), \
        "no builder for the encoder pretraining corpus; without it the encoder cannot be retrained"


@pytest.mark.parametrize(
    "cache", ["afdb", "cath_backbone_split"], ids=lambda s: s
)
def test_the_documentation_says_how_to_build_each_cache(cache: str):
    data = (ROOT / "docs" / "DATA.md").read_text()
    assert f"`{cache}`" in data, f"docs/DATA.md does not mention the {cache} cache"
    builder = "build_afdb_cache.py" if cache == "afdb" else "build_cache.py"
    assert builder in data, f"docs/DATA.md does not say to run {builder}"


def test_the_readme_is_honest_about_what_a_cpu_install_cannot_do():
    """The released encoders need mamba_ssm, which the plain install does not provide."""
    readme = (ROOT / "README.md").read_text()
    assert "mamba_ssm" in readme or "mamba-ssm" in readme
    assert "cannot run the released encoders" in readme, \
        "the README must say the plain install cannot run the released weights"


def test_active_weights_match_the_default_encoder():
    from psjepa import sota
    with WEIGHTS_MANIFEST.open() as f:
        rows = list(csv.DictReader((line for line in f if not line.startswith("#")), delimiter="\t"))
    assert [row["path"] for row in rows] == [str(sota.SOTA_ENCODER_WEIGHTS)]
    assert sota.SOTA_ENCODER_STEPS == 35000
    assert sota.SOTA_ENCODER_BATCH_SIZE * sota.SOTA_ENCODER_ACCUMULATION == 512


def test_distribution_excludes_private_research_and_manuscript_artifacts():
    excluded = ("docs/research/", "docs/analysis/", "docs/experiments/",
                "docs/final_draft/", "docs/_archive/", "notebooks/")
    assert not [p for p in _tracked() if p.startswith(excluded)]


def test_distribution_includes_weight_and_upstream_license_texts():
    tracked = _tracked()
    for path in ("weights/LICENSE",
                 "LICENSES/StructTokenBench-MIT.txt", "LICENSES/Apache-2.0.txt"):
        assert path in tracked
        assert (ROOT / path).stat().st_size > 500
    assert "MIT" in (ROOT / "weights/LICENSE").read_text()


def test_published_benchmark_cells_and_category_means():
    import statistics
    with (ROOT / "docs/benchmark_results.csv").open() as f:
        cells = list(csv.DictReader(f))
    assert len(cells) == len({(r["task"],r["split"]) for r in cells}) == 24
    groups = [
        [float(r["value"]) for r in cells if r["task"] != "Homo" and not r["task"].startswith("Flex")],
        [float(r["value"]) for r in cells if r["task"].startswith("Flex")],
        [float(r["value"]) for r in cells if r["task"] == "Homo"],
    ]
    assert [len(g) for g in groups] == [15,6,3]
    means = [statistics.mean(g) for g in groups]
    for doc in ["README.md"]:
        text = (ROOT/doc).read_text()
        for value in [*means, statistics.mean(means)]:
            assert f"{value:.2f}" in text
