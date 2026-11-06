#!/usr/bin/env python3
'Evaluate backbone designability with ProteinMPNN and ESMFold.'
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from psjepa.data.reference import REFERENCE_CACHE, REFERENCE_TRAIN_SPLIT
from psjepa.paths import project_root

MPNN_DIR = Path(
    os.environ.get("PROTEINMPNN_DIR")
    or project_root() / "external" / "ProteinMPNN"
)
BACKBONE_ATOMS = ("N", "CA", "C", "O")
FOLD_FIELDS = (
    "arm", "name", "sample", "length", "sc_rmsd", "sc_tm", "plddt",
    "motif_rmsd", "motif_residues", "mpnn_score", "seq_recovery", "sequence",
)


# ---------------------------------------------------------------- PDB reading
def read_backbone_atoms(path: Path, atoms=BACKBONE_ATOMS) -> np.ndarray:
    """Backbone atoms per residue, in file order, shaped [L, len(atoms), 3].

    The MotifBench text names N/CA/C, but its executable Scaffold-Lab evaluator selects
    ``backbone``, which is N/CA/C/O. We follow the executable protocol so this number can be
    compared with leaderboard evaluations.
    """
    wanted = {name: i for i, name in enumerate(atoms)}
    residues: dict[tuple, np.ndarray] = {}
    chain = None
    for line in path.read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        name = line[12:16].strip()
        if name not in wanted or line[16] not in (" ", "A"):
            continue
        if chain is None:
            chain = line[21]
        elif line[21] != chain:
            break
        key = (line[21], line[22:27])
        residues.setdefault(key, np.full((len(atoms), 3), np.nan))[wanted[name]] = [
            float(line[30:38]), float(line[38:46]), float(line[46:54])]
    if not residues:
        raise ValueError(f"no backbone atoms in {path}")
    return np.stack(list(residues.values()))


def read_backbone(path: Path) -> np.ndarray:
    """CA coordinates of a single-chain backbone PDB, in file order.

    Only the first chain is read and only the first alternate location of each atom, so a
    multi-chain or altloc-carrying file cannot silently merge into one longer trace.
    """
    coordinates: list[np.ndarray] = []
    seen: set[tuple[str, str]] = set()
    chain: str | None = None
    for line in path.read_text().splitlines():
        if not line.startswith("ATOM") or line[12:16].strip() != "CA":
            continue
        if line[16] not in (" ", "A"):          # alternate location
            continue
        if chain is None:
            chain = line[21]
        elif line[21] != chain:
            break
        key = (line[21], line[22:27])
        if key in seen:
            continue
        seen.add(key)
        coordinates.append(np.array(
            [float(line[30:38]), float(line[38:46]), float(line[46:54])]
        ))
    if not coordinates:
        raise ValueError(f"no CA atoms in {path}")
    return np.stack(coordinates)


THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def read_residue_names(path: Path) -> list[str]:
    """Residue names in file order, one per residue of the first chain."""
    names: dict[str, str] = {}
    chain = None
    for line in path.read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        if chain is None:
            chain = line[21]
        elif line[21] != chain:
            break
        names.setdefault(line[22:27], line[17:20].strip())
    return list(names.values())


def fixed_motif_positions(pdb: Path) -> list[int]:
    """1-based positions whose amino acid the sequence designer may not choose.

    MotifBench constrains the motif's amino acid types except at the residues it lists in
    `redesign_idcs`, which it marks `UNK` and strips to backbone atoms. Function is carried by
    side chains, so a design that is free to put anything at the site is solving an easier
    problem than the benchmark poses. A motif residue that is natively glycine is still fixed --
    the placeholder scaffold is glycine too, so the motif list, not the name, decides.
    """
    motif = read_motif(pdb)
    if motif is None:
        return []
    names = read_residue_names(pdb)
    held = [int(i) + 1 for i in motif
            if 0 <= i < len(names) and names[i] in THREE_TO_ONE]


    if held and all(names[int(i)] == "GLY" for i in motif):
        raise ValueError(
            f"{pdb.name} has no motif sequence to hold: every motif residue is GLY, so this "
            "design predates the residue names being carried through. Regenerate it, or score "
            "without --fix-motif-sequence.")
    return held


def read_motif(path: Path) -> np.ndarray | None:
    """Residue indices (0-based) that were handed to the generator, from the PDB header.

    The optional header is `REMARK 900 <arm>; motif residues 17,18,...` (1-based, in the
    written chain's own numbering). A file without that line has no motif and is scored the
    ordinary way.
    """
    for line in path.read_text().splitlines():
        if line.startswith("ATOM"):
            break
        if "motif residues" in line:
            field = line.split("motif residues", 1)[1].strip()
            if not field:
                return None
            return np.array([int(x) - 1 for x in field.split(",") if x.strip()], dtype=int)
    return None


def read_motif_reference(path: Path) -> Path | None:
    """The file the motif must be scored against, if the design names one.

    A design that had the motif's coordinates written into it is its own reference: the site is
    exact by construction. A design that received the motif only as a latent condition is not --
    the whole question is whether the model put the site where it belongs -- so the generator
    writes `REMARK 901 motif reference <relative path>` pointing at the benchmark's own segments,
    placed at the residue numbers the design used. Scoring against the design itself in that case
    would measure self-consistency and silently pass a wrong site.
    """
    for line in path.read_text().splitlines():
        if line.startswith("ATOM"):
            break
        if "motif reference" in line:
            field = line.split("motif reference", 1)[1].strip()
            if not field:
                return None
            resolved = (path.parent / field).resolve()
            return resolved if resolved.exists() else None
    return None


def read_backbone_by_number(path: Path, atoms=BACKBONE_ATOMS) -> dict[int, np.ndarray]:
    """Backbone atoms keyed by residue number rather than file order.

    A motif-only reference holds just the site's residues, so its file order is 0..len(motif)-1
    while the indices that name the motif belong to the full-length design. Keying on the number
    written in the file lets the two be lined up.
    """
    wanted = {name: i for i, name in enumerate(atoms)}
    residues: dict[int, np.ndarray] = {}
    for line in path.read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        name = line[12:16].strip()
        if name not in wanted or line[16] not in (" ", "A"):
            continue
        number = int(line[22:26])
        residues.setdefault(number, np.full((len(atoms), 3), np.nan))[wanted[name]] = [
            float(line[30:38]), float(line[38:46]), float(line[46:54])]
    return residues


def motif_rmsd_backbone(prediction_path: Path, reference_path: Path, motif: np.ndarray,
                        reference_by_number: bool = False) -> float:
    """Motif RMSD over N/CA/C/O, as computed by MotifBench's Scaffold-Lab evaluator."""
    p = read_backbone_atoms(prediction_path)
    if reference_by_number:
        numbered = read_backbone_by_number(reference_path)
        keep = np.array([i for i in motif if 0 <= i < len(p) and (i + 1) in numbered], dtype=int)
        if len(keep) < 3:
            return float("nan")
        a = p[keep].reshape(-1, 3)
        b = np.stack([numbered[i + 1] for i in keep]).reshape(-1, 3)
        good = np.isfinite(a).all(-1) & np.isfinite(b).all(-1)
        if good.sum() < 12:
            return float("nan")
        return kabsch_rmsd(a[good], b[good])
    q = read_backbone_atoms(reference_path)
    keep = motif[(motif >= 0) & (motif < min(len(p), len(q)))]
    if len(keep) < 3:
        return float("nan")
    a = p[keep].reshape(-1, 3)
    b = q[keep].reshape(-1, 3)
    good = np.isfinite(a).all(-1) & np.isfinite(b).all(-1)
    if good.sum() < 12:
        return float("nan")
    return kabsch_rmsd(a[good], b[good])


def motif_rmsd(prediction: np.ndarray, reference: np.ndarray, motif: np.ndarray) -> float:
    """CA RMSD over the motif alone, after superposing on the motif alone.

    This is the number that decides whether a scaffold "kept" its site: the designed
    sequence is folded by ESMFold with no knowledge of the constraint, so the motif has to
    come back on its own. The convention is a 1 A threshold, with the whole-design scRMSD
    at 2 A; both must hold for the design to count.
    """
    keep = motif[(motif >= 0) & (motif < min(len(prediction), len(reference)))]
    if len(keep) < 3:
        return float("nan")
    return kabsch_rmsd(prediction[keep], reference[keep])


def kabsch_rmsd(prediction: np.ndarray, target: np.ndarray) -> float:
    p = prediction - prediction.mean(axis=0, keepdims=True)
    q = target - target.mean(axis=0, keepdims=True)
    u, _, vh = np.linalg.svd(p.T @ q)
    rotation = u @ vh
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vh
    return float(np.sqrt(((p @ rotation - q) ** 2).sum(axis=-1).mean()))


# ------------------------------------------------------------- stage 1: MPNN
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def remove_owned_path(path: Path) -> None:
    """Remove one known evaluation artifact; never accepts a broad run root."""
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def validate_output_root(path: Path) -> None:
    resolved = path.resolve()
    protected = {Path("/").resolve(), Path.home().resolve(), Path(__file__).resolve().parents[1]}
    if resolved in protected:
        raise SystemExit(f"refusing to use protected directory as --out: {resolved}")


def run_mpnn(pdbs: list[Path], out_dir: Path, args) -> Path:
    """Design `--num-seq` sequences for every PDB; returns the directory of FASTA files."""


    out_dir = out_dir.resolve()
    seqs = out_dir / "seqs"
    stamp = out_dir / "mpnn_settings.json"


    fixed = {pdb.stem: {"A": fixed_motif_positions(pdb)} for pdb in pdbs} if args.fix_motif_sequence else {}


    settings = {
        "num_seq": args.num_seq, "temperature": args.temperature,
        "backbone_noise": args.backbone_noise, "model": args.mpnn_model,
        "seed": args.mpnn_seed,
        "pdb_sha256": {pdb.name: sha256_file(pdb) for pdb in pdbs},
        "fixed_motif_positions": fixed,
    }
    cached = json.loads(stamp.read_text()) if stamp.exists() else None
    expected_fastas = {f"{pdb.stem}.fa" for pdb in pdbs}
    present_fastas = {path.name for path in seqs.glob("*.fa")} if seqs.exists() else set()
    if not args.force and cached == settings and present_fastas == expected_fastas:
        print(f"[skip] ProteinMPNN already ran for {out_dir} with these settings", flush=True)
        return seqs
    if not args.force and (cached is not None or present_fastas):
        raise RuntimeError(
            f"ProteinMPNN cache at {out_dir} does not match the current PDBs/settings; "
            "use a new --out directory or pass --force")
    if args.force:
        for artifact in (seqs, out_dir / "inputs", out_dir / "pdbs.jsonl",
                         out_dir / "fixed_positions.jsonl", stamp):
            remove_owned_path(artifact)
    staging = out_dir / "inputs"
    staging.mkdir(parents=True, exist_ok=True)
    for pdb in pdbs:
        link = staging / pdb.name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(pdb.resolve())
    jsonl = out_dir / "pdbs.jsonl"
    python = sys.executable
    design = [
         "--jsonl_path", str(jsonl), "--out_folder", str(out_dir),
         "--num_seq_per_target", str(args.num_seq),
         "--sampling_temp", str(args.temperature),
         "--backbone_noise", str(args.backbone_noise),
         "--model_name", args.mpnn_model,
         "--seed", str(args.mpnn_seed),
         "--batch_size", str(min(args.num_seq, 8))]
    if any(v["A"] for v in fixed.values()):
        fixed_path = out_dir / "fixed_positions.jsonl"
        fixed_path.write_text(json.dumps(fixed) + "\n")
        design += ["--fixed_positions_jsonl", str(fixed_path)]
        print(f"[mpnn] motif sequence held at "
              f"{sum(len(v['A']) for v in fixed.values())} positions over "
              f"{sum(1 for v in fixed.values() if v['A'])} of {len(fixed)} backbones",
              flush=True)
    commands = [
        [python, str(MPNN_DIR / "helper_scripts" / "parse_multiple_chains.py"),
         f"--input_path={staging}", f"--output_path={jsonl}"],
        [python, str(MPNN_DIR / "protein_mpnn_run.py")] + design,
    ]
    for command in commands:
        result = subprocess.run(command, cwd=MPNN_DIR)
        if result.returncode != 0:
            raise RuntimeError(f"ProteinMPNN failed for {out_dir} ({command[1]})")
    present_fastas = {path.name for path in seqs.glob("*.fa")} if seqs.exists() else set()
    if present_fastas != expected_fastas:
        raise RuntimeError(
            f"ProteinMPNN produced the wrong FASTA set for {out_dir}: "
            f"missing={sorted(expected_fastas - present_fastas)[:8]}, "
            f"extra={sorted(present_fastas - expected_fastas)[:8]}")
    stamp.write_text(json.dumps(settings, indent=2))
    return seqs


def read_designs(path: Path) -> list[dict]:
    """Designed sequences from a ProteinMPNN FASTA (entry 0 is the input sequence)."""
    entries, header, buffer = [], None, []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if header is not None:
                entries.append((header, "".join(buffer)))
            header, buffer = line[1:], []
        elif line.strip():
            buffer.append(line.strip())
    if header is not None:
        entries.append((header, "".join(buffer)))
    designs = []
    for header, sequence in entries[1:]:
        fields = {}
        for item in header.split(","):
            if "=" in item:
                key, value = item.split("=", 1)
                fields[key.strip()] = value.strip()
        designs.append({
            "sequence": sequence.replace("/", ""),
            "mpnn_score": float(fields.get("score", "nan")),
            "seq_recovery": float(fields.get("seq_recovery", "nan")),
            "sample": int(fields.get("sample", len(designs) + 1)),
        })
    return designs


def validate_fixed_motif_sequence(pdb: Path, designs: list[dict]) -> None:
    names = read_residue_names(pdb)
    fixed = fixed_motif_positions(pdb)
    expected = {position: THREE_TO_ONE[names[position - 1]] for position in fixed}
    for design in designs:
        sequence = design["sequence"]
        wrong = [(position, residue, sequence[position - 1] if position <= len(sequence) else None)
                 for position, residue in expected.items()
                 if position > len(sequence) or sequence[position - 1] != residue]
        if wrong:
            raise RuntimeError(
                f"ProteinMPNN changed fixed motif residues in {pdb.name}, "
                f"sample {design['sample']}: {wrong[:8]}")


# ---------------------------------------------------------- stage 2: ESMFold
def load_esmfold(args):
    if args.esmfold_backend == "fair-esm":
        import esm

        pretrained = getattr(esm, "pretrained", None)
        if pretrained is None or not hasattr(pretrained, "esmfold_v1"):
            raise RuntimeError(
                "--esmfold-backend fair-esm requires the legacy fair-esm package with "
                "esm.pretrained.esmfold_v1(); the installed `esm` package is not compatible")
        model = pretrained.esmfold_v1().eval()
        model = model.cuda() if torch.cuda.is_available() else model.float()
        return "fair-esm", model

    from transformers import EsmForProteinFolding

    # This backend is retained for generic diagnostics. Official MotifBench mode requires the
    # original fair-esm implementation used by Scaffold-Lab.
    model = EsmForProteinFolding.from_pretrained(args.esmfold_model)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        model.esm = model.esm.half()
        model.trunk.set_chunk_size(args.chunk_size)
    return "transformers", model


def ca_and_plddt_from_pdb(text: str) -> tuple[np.ndarray, float]:
    coordinates = []
    confidence = []
    chain = None
    seen = set()
    for line in text.splitlines():
        if not line.startswith("ATOM") or line[12:16].strip() != "CA":
            continue
        if line[16] not in (" ", "A"):
            continue
        if chain is None:
            chain = line[21]
        elif line[21] != chain:
            break
        key = (line[21], line[22:27])
        if key in seen:
            continue
        seen.add(key)
        coordinates.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        confidence.append(float(line[60:66]))
    if not coordinates:
        raise ValueError("ESMFold returned no CA atoms")
    return np.asarray(coordinates), float(np.mean(confidence))


@torch.no_grad()
def fold(model, sequence: str, pdb_path: Path | None = None):
    """CA coordinates and mean CA pLDDT of one sequence, optionally written as a PDB."""
    backend, model = model
    if backend == "fair-esm":
        output = model.infer(sequence)
        pdb_text = model.output_to_pdb(output)[0]
        if pdb_path is not None:
            pdb_path.parent.mkdir(parents=True, exist_ok=True)
            pdb_path.write_text(pdb_text)
        return ca_and_plddt_from_pdb(pdb_text)

    from transformers.models.esm.openfold_utils.feats import atom14_to_atom37

    output = model.infer([sequence])
    positions = atom14_to_atom37(output["positions"][-1], output)[0]   # [L, 37, 3]
    ca = positions[:, 1].float().cpu().numpy()
    plddt = float(output["plddt"][0, :, 1].float().mean())
    # transformers returns pLDDT on 0-1 (the reference ESM repo scales it to 0-100); accept either.
    plddt = plddt * 100.0 if plddt <= 1.5 else plddt
    if pdb_path is not None:
        pdb_path.parent.mkdir(parents=True, exist_ok=True)
        # transformers writes pLDDT straight into the B-factor column; scale it to the 0-100
        # range every structure viewer expects.
        written = dict(output)
        written["plddt"] = output["plddt"] * (100.0 if float(output["plddt"].max()) <= 1.5 else 1.0)
        pdb_path.write_text(model.output_to_pdb(written)[0])
    return ca, plddt


def tm_score(prediction: np.ndarray, reference: np.ndarray, sequence: str) -> float:
    """TM-score of the prediction against the generated backbone, normalised by the latter."""
    from tmtools import tm_align

    reference_sequence = sequence[: len(reference)]
    result = tm_align(
        prediction.astype(np.float64), reference.astype(np.float64),
        sequence[: len(prediction)], reference_sequence,
    )
    return float(result.tm_norm_chain2)


# --------------------------------------------------------------- aggregation
def pairwise_tm(structures: list[tuple[np.ndarray, str]]) -> float:
    from tmtools import tm_align

    values = []
    for i in range(len(structures)):
        for j in range(i + 1, len(structures)):
            (a, sa), (b, sb) = structures[i], structures[j]
            values.append(float(tm_align(a.astype(np.float64), b.astype(np.float64),
                                         sa[: len(a)], sb[: len(b)]).tm_norm_chain1))
    return float(np.mean(values)) if values else float("nan")


def novelty(structures: list[tuple[np.ndarray, str]], references: list[np.ndarray]) -> list[float]:
    from tmtools import tm_align

    poly = "A" * 4096
    out = []
    for coordinates, _ in structures:
        best = 0.0
        for reference in references:
            value = float(tm_align(
                coordinates.astype(np.float64), reference.astype(np.float64),
                poly[: len(coordinates)], poly[: len(reference)],
            ).tm_norm_chain1)
            best = max(best, value)
        out.append(best)
    return out


def reference_chains(cache: str, split: str, count: int, max_residues: int, seed: int):
    """CA traces of real chains, for the novelty term."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from psjepa.data import BackboneDataset

    dataset = BackboneDataset(cache, split, max_residues=max_residues, augment=False)
    rng = np.random.RandomState(seed)
    index = rng.choice(len(dataset), size=min(count, len(dataset)), replace=False)
    out = []
    for i in index:
        item = dataset[int(i)]
        coordinates = item["coords"]
        mask = item["residue_mask"].bool() if torch.is_tensor(item["residue_mask"]) else None
        trace = coordinates[mask, 1] if mask is not None else coordinates[:, 1]
        out.append(np.asarray(trace, dtype=np.float64))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", help="sampling output directory holding pdb/<arm>/*.pdb")
    parser.add_argument("--pdb-dir", help="a single flat directory of PDBs (use with --arm-name)")
    parser.add_argument("--arm-name", default="samples")
    parser.add_argument("--arms", default="prior,real,data")
    parser.add_argument("--out", default=None)
    parser.add_argument("--num-seq", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--backbone-noise", type=float, default=0.0)
    parser.add_argument("--mpnn-model", default="v_48_020")
    parser.add_argument("--mpnn-seed", type=int, default=37)
    parser.add_argument("--limit", type=int, default=None, help="PDBs per arm; default is all")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--esmfold-model", default="facebook/esmfold_v1")
    parser.add_argument("--esmfold-backend", choices=["transformers", "fair-esm"], default=None,
                        help="fair-esm is the implementation used by official MotifBench")
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--rmsd-threshold", type=float, default=2.0)
    parser.add_argument("--fix-motif-sequence", action="store_true",
                        help="hold the motif's amino acid types during sequence design, except "
                             "at the positions the benchmark marks UNK. Required for a MotifBench "
                             "number; leave off for a backbone-only comparison.")
    parser.add_argument("--motif-threshold", type=float, default=1.0,
                        help="motif backbone RMSD below which the refolded site counts as kept")
    parser.add_argument("--tm-threshold", type=float, default=0.5)
    parser.add_argument("--write-folded", type=int, default=8,
                        help="chains per arm whose folded structures are all written out")
    parser.add_argument("--min-length", type=int, default=20)
    parser.add_argument("--novelty-refs", type=int, default=0)
    # The novelty reference must come from the same distribution the generator was
    # trained on, or a low max-TM only says the two datasets differ. Defaults follow
    # psjepa.data.reference; pass these explicitly when scoring an older CATH run.
    parser.add_argument("--novelty-cache", default=REFERENCE_CACHE)
    parser.add_argument("--novelty-split", default=REFERENCE_TRAIN_SPLIT)
    parser.add_argument("--mpnn-only", action="store_true", help="stop after the design stage")
    parser.add_argument("--motifbench", action="store_true",
                        help="enforce the complete official 30x100, 8-sequence protocol")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.esmfold_backend = args.esmfold_backend or (
        "fair-esm" if args.motifbench else "transformers")
    if args.motifbench:
        args.fix_motif_sequence = True
        required = {
            "num_seq": (args.num_seq, 8),
            "temperature": (args.temperature, 0.1),
            "backbone_noise": (args.backbone_noise, 0.0),
            "mpnn_model": (args.mpnn_model, "v_48_020"),
            "rmsd_threshold": (args.rmsd_threshold, 2.0),
            "motif_threshold": (args.motif_threshold, 1.0),
            "esmfold_backend": (args.esmfold_backend, "fair-esm"),
        }
        wrong = {key: value for key, value in required.items() if value[0] != value[1]}
        if wrong:
            raise SystemExit(f"non-official settings passed with --motifbench: {wrong}")
        if not args.pdb_dir or args.samples:
            raise SystemExit("--motifbench requires one flat --pdb-dir and does not accept --samples")
        if args.limit not in (None, 3000):
            raise SystemExit("official MotifBench requires all 3000 backbones; omit --limit")

    if not MPNN_DIR.exists():
        raise SystemExit(f"ProteinMPNN not found at {MPNN_DIR}; set PROTEINMPNN_DIR")
    if args.esmfold_backend == "fair-esm" and not args.mpnn_only:
        import esm
        if not hasattr(getattr(esm, "pretrained", None), "esmfold_v1"):
            raise SystemExit(
                "official MotifBench needs legacy fair-esm (esm.pretrained.esmfold_v1); "
                "select a Python environment that provides it")

    if args.pdb_dir:
        if args.arm_name in ("", ".", "..") or Path(args.arm_name).name != args.arm_name:
            raise SystemExit("--arm-name must be one safe path component")
        arms = {args.arm_name: Path(args.pdb_dir)}
        root = Path(args.out or Path(args.pdb_dir).parent / "designability")
    else:
        if not args.samples:
            raise SystemExit("pass --samples or --pdb-dir")
        base = Path(args.samples)
        arms = {}
        for arm in args.arms.split(","):
            if arm in ("", ".", "..") or Path(arm).name != arm:
                raise SystemExit(f"unsafe arm name: {arm!r}")
            directory = base / "pdb" / arm
            if directory.is_dir() and list(directory.glob("*.pdb")):
                arms[arm] = directory
            else:
                print(f"[warn] no PDBs for arm {arm} under {base/'pdb'}", flush=True)
        root = Path(args.out or base / "designability")
    if not arms:
        raise SystemExit("no arm has PDBs to score")
    validate_output_root(root)
    root.mkdir(parents=True, exist_ok=True)

    # -- stage 1: design ----------------------------------------------------
    work: dict[str, list[tuple[Path, list[dict]]]] = {}
    for arm, directory in arms.items():
        usable = []
        for candidate in sorted(directory.glob("*.pdb")):
            try:
                length = len(read_backbone(candidate))
            except ValueError as error:
                print(f"[warn] {error}", flush=True)
                continue
            if args.min_length <= length <= args.max_length:
                usable.append(candidate)

        pdbs = usable if args.limit is None else usable[: args.limit]
        if args.motifbench:
            if len(pdbs) != 3000:
                raise SystemExit(
                    f"official MotifBench requires 3000 usable PDBs, found {len(pdbs)} in {directory}")
            without_motif = [pdb.name for pdb in pdbs if read_motif(pdb) is None]
            if without_motif:
                raise SystemExit(f"MotifBench PDBs without motif metadata: {without_motif[:8]}")
        out_dir = root / arm
        out_dir.mkdir(parents=True, exist_ok=True)
        seqs = run_mpnn(pdbs, out_dir, args)
        work[arm] = []
        for pdb in pdbs:
            fasta = seqs / f"{pdb.stem}.fa"
            if not fasta.exists():
                print(f"[warn] no designs for {pdb.name}", flush=True)
                continue
            designs = read_designs(fasta)
            if not designs:
                print(f"[warn] empty design file for {pdb.name}", flush=True)
                continue
            if args.motifbench and len(designs) != 8:
                raise RuntimeError(
                    f"{pdb.name} has {len(designs)} ProteinMPNN sequences; official protocol needs 8")
            if args.fix_motif_sequence:
                validate_fixed_motif_sequence(pdb, designs)
            work[arm].append((pdb, designs))
        print(f"[mpnn] {arm}: {len(work[arm])} backbones x {args.num_seq} sequences", flush=True)
    if args.mpnn_only:
        return

    # -- stage 2: fold ------------------------------------------------------
    folds_path = root / "folds.csv"
    fold_stamp = root / "fold_settings.json"
    fold_settings = {
        "esmfold_backend": args.esmfold_backend,
        "esmfold_model": args.esmfold_model,
        "chunk_size": args.chunk_size,
        "inputs": {
            arm: {
                pdb.name: {
                    "pdb_sha256": sha256_file(pdb),
                    "motif_reference_sha256": (
                        sha256_file(reference) if (reference := read_motif_reference(pdb)) else None),
                    "sequences": [
                        {"sample": design["sample"], "sequence": design["sequence"]}
                        for design in designs
                    ],
                }
                for pdb, designs in items
            }
            for arm, items in work.items()
        },
    }
    cached_fold_settings = json.loads(fold_stamp.read_text()) if fold_stamp.exists() else None
    if folds_path.exists() and not args.force and cached_fold_settings != fold_settings:
        raise RuntimeError(
            f"fold cache at {root} does not match the current backbones/sequences/model; "
            "use a new --out directory or pass --force")
    done: set[tuple[str, str, int]] = set()
    rows: list[dict] = []
    if args.force:
        remove_owned_path(folds_path)
        remove_owned_path(fold_stamp)
        for arm in work:
            remove_owned_path(root / arm / "folded")
    fold_stamp.write_text(json.dumps(fold_settings, indent=2))
    if folds_path.exists():
        with folds_path.open() as handle:
            for row in csv.DictReader(handle):
                key = (row["arm"], row["name"], int(row["sample"]))
                if key in done:
                    raise RuntimeError(f"duplicate fold cache row: {key}")
                rows.append(row)
                done.add(key)
        print(f"[resume] {len(done)} sequences already folded", flush=True)
    pending = [
        (arm, pdb, design)
        for arm, items in work.items()
        for pdb, designs in items
        for design in designs
        if (arm, pdb.stem, design["sample"]) not in done
    ]
    if pending:
        model = load_esmfold(args)
        handle = folds_path.open("a", newline="")
        writer = csv.DictWriter(handle, fieldnames=list(FOLD_FIELDS))
        if not done:
            writer.writeheader()
        pending.sort(key=lambda item: len(item[2]["sequence"]))
        # Keep the folded structures of a few chains per arm, for figures and manual inspection.
        keep_pdb = {
            arm: {pdb.stem for pdb, _ in items[: args.write_folded]}
            for arm, items in work.items()
        }
        for index, (arm, pdb, design) in enumerate(pending, start=1):
            reference = read_backbone(pdb)
            raw_sequence = design["sequence"]
            if args.motifbench and len(raw_sequence) != len(reference):
                raise RuntimeError(
                    f"ProteinMPNN sequence length mismatch for {pdb.stem}, "
                    f"sample {design['sample']}: sequence={len(raw_sequence)}, "
                    f"backbone={len(reference)}")
            sequence = raw_sequence[: len(reference)]
            # The executable MotifBench evaluator scores N/CA/C/O, which needs the predicted
            # structure on disk. With a motif every fold is temporarily written.
            motif = read_motif(pdb)
            keep_file = pdb.stem in keep_pdb.get(arm, ())
            folded_path = (root / arm / "folded" / f"{pdb.stem}_seq{design['sample']}.pdb"
                           if (keep_file or motif is not None) else None)
            try:
                prediction, plddt = fold(model, sequence, folded_path)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if args.motifbench:
                    raise RuntimeError(
                        f"ESMFold failed in official mode on {arm}/{pdb.stem} "
                        f"sample {design['sample']} L={len(sequence)}") from error
                print(f"[skip] ESMFold failed on {arm}/{pdb.stem} sample {design['sample']} "
                      f"L={len(sequence)}: {type(error).__name__}: {error}", flush=True)
                continue
            if args.motifbench and len(prediction) != len(reference):
                raise RuntimeError(
                    f"ESMFold length mismatch for {pdb.stem}, sample {design['sample']}: "
                    f"prediction={len(prediction)}, backbone={len(reference)}")
            limit = min(len(prediction), len(reference))
            given = ""
            if motif is not None:
                try:
                    reference_pdb = read_motif_reference(pdb)
                    given = round(motif_rmsd_backbone(
                        folded_path, reference_pdb or pdb, motif,
                        reference_by_number=reference_pdb is not None), 4)
                except Exception as error:      # fall back to the CA-only number, and say so
                    if args.motifbench:
                        raise RuntimeError(
                            f"official motif RMSD failed for {pdb.stem}; refusing CA fallback") from error
                    print(f"[warn] backbone motif RMSD failed for {pdb.stem}: {error}", flush=True)
                    given = round(motif_rmsd(prediction, reference, motif), 4)
                if not keep_file and folded_path is not None and folded_path.exists():
                    folded_path.unlink()
            row = {
                "arm": arm, "name": pdb.stem, "sample": design["sample"],
                "length": len(reference),
                "sc_rmsd": round(kabsch_rmsd(prediction[:limit], reference[:limit]), 4),
                "sc_tm": round(tm_score(prediction[:limit], reference[:limit], sequence), 4),
                "plddt": round(plddt, 2),
                "motif_rmsd": given,
                "motif_residues": 0 if motif is None else len(motif),
                "mpnn_score": design["mpnn_score"],
                "seq_recovery": design["seq_recovery"],
                "sequence": sequence,
            }
            writer.writerow(row)
            handle.flush()
            rows.append(row)
            if index % 25 == 0 or index == len(pending):
                print(f"[esmfold] {index}/{len(pending)} last={arm}/{pdb.stem} "
                      f"scRMSD={row['sc_rmsd']} scTM={row['sc_tm']} pLDDT={row['plddt']}", flush=True)
        handle.close()
        del model
        torch.cuda.empty_cache()

    if args.motifbench:
        expected_fold_keys = {
            (arm, pdb.stem, design["sample"])
            for arm, items in work.items()
            for pdb, designs in items
            for design in designs
        }
        actual_fold_keys = {
            (row["arm"], row["name"], int(row["sample"])) for row in rows
        }
        if actual_fold_keys != expected_fold_keys:
            raise RuntimeError(
                "official MotifBench refolding is incomplete: "
                f"missing={len(expected_fold_keys - actual_fold_keys)}, "
                f"extra={len(actual_fold_keys - expected_fold_keys)}")

    # -- aggregate ----------------------------------------------------------
    by_chain: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        row = {k: (float(v) if k in ("sc_rmsd", "sc_tm", "plddt", "mpnn_score", "seq_recovery")
                   and v not in ("", None) else v) for k, v in row.items()}
        row["motif_rmsd"] = (float(row["motif_rmsd"])
                             if str(row.get("motif_rmsd", "")).strip() not in ("", "nan")
                             else float("nan"))
        by_chain.setdefault((row["arm"], row["name"]), []).append(row)
    best_rows = []
    for (arm, name), group in sorted(by_chain.items()):
        has_motif = any(np.isfinite(r["motif_rmsd"]) for r in group)
        if has_motif:


            passing = [r for r in group
                       if np.isfinite(r["motif_rmsd"]) and r["motif_rmsd"] < args.motif_threshold]
            best = min(passing or group, key=lambda r: r["sc_rmsd"])
        else:
            best = min(group, key=lambda r: r["sc_rmsd"])
        best_rows.append({
            "arm": arm, "name": name, "length": int(float(group[0]["length"])),
            "best_sc_rmsd": best["sc_rmsd"],
            "motif_rmsd_of_best": best["motif_rmsd"],
            "best_motif_rmsd": (min((r["motif_rmsd"] for r in group if np.isfinite(r["motif_rmsd"])),
                                    default=float("nan"))),
            "motif_success": bool(has_motif and best["sc_rmsd"] < args.rmsd_threshold
                                  and np.isfinite(best["motif_rmsd"])
                                  and best["motif_rmsd"] < args.motif_threshold),
            "best_sc_tm": max(r["sc_tm"] for r in group),
            "plddt_of_best": best["plddt"],
            "mean_plddt": float(np.mean([r["plddt"] for r in group])),
            "mean_mpnn_score": float(np.mean([r["mpnn_score"] for r in group])),
            "sequences": len(group),
            "sequence_of_best": best["sequence"],
        })
    if not best_rows:
        raise SystemExit("no sequence was folded successfully; nothing to aggregate")
    with (root / "best_per_chain.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(best_rows[0].keys()))
        writer.writeheader()
        writer.writerows(best_rows)

    references = (
        reference_chains(args.novelty_cache, args.novelty_split, args.novelty_refs,
                         args.max_length, args.mpnn_seed)
        if args.novelty_refs > 0 else []
    )
    summary = {}
    for arm in arms:
        group = [r for r in best_rows if r["arm"] == arm]
        if not group:
            continue
        rmsd = np.array([r["best_sc_rmsd"] for r in group])
        tm = np.array([r["best_sc_tm"] for r in group])
        designable = [r for r in group if r["best_sc_rmsd"] < args.rmsd_threshold]


        basis = designable if len(designable) >= 2 else group
        structures = [
            (read_backbone(arms[arm] / f"{r['name']}.pdb"), r["sequence_of_best"])
            for r in basis
        ]
        motif_rows = [r for r in group if np.isfinite(r.get("best_motif_rmsd", float("nan")))]
        entry = {
            "n_chains": len(group),
            "designable_fraction": float((rmsd < args.rmsd_threshold).mean()),
            "sc_tm_pass_fraction": float((tm > args.tm_threshold).mean()),
            "sc_rmsd_mean": float(rmsd.mean()),
            "sc_rmsd_median": float(np.median(rmsd)),
            "sc_tm_mean": float(tm.mean()),
            "plddt_mean": float(np.mean([r["mean_plddt"] for r in group])),
            "plddt_of_best_mean": float(np.mean([r["plddt_of_best"] for r in group])),
            "mpnn_score_mean": float(np.mean([r["mean_mpnn_score"] for r in group])),
            "length_mean": float(np.mean([r["length"] for r in group])),
            "diversity_pairwise_tm": pairwise_tm(structures[:48]),
            "diversity_basis": "designable" if basis is designable else "all samples",
            "n_designable": len(designable),
        }
        if motif_rows:
            # The headline number for motif scaffolding: scRMSD < 2 A *and* motif < 1 A.
            entry["motif_success_fraction"] = float(
                np.mean([1.0 if r["motif_success"] else 0.0 for r in motif_rows]))
            entry["motif_rmsd_median"] = float(np.median([r["best_motif_rmsd"] for r in motif_rows]))
            entry["n_motif_chains"] = len(motif_rows)
        if references and structures:
            values = novelty(structures[:48], references)
            entry["novelty_max_tm_mean"] = float(np.mean(values))
        summary[arm] = entry
        print(f"[{arm}] designable {entry['designable_fraction']:.3f} "
              f"scRMSD median {entry['sc_rmsd_median']:.2f} A  scTM {entry['sc_tm_mean']:.3f} "
              f"pLDDT {entry['plddt_mean']:.1f}", flush=True)

    def finite(value):
        if isinstance(value, float) and not np.isfinite(value):
            return None
        if isinstance(value, dict):
            return {k: finite(v) for k, v in value.items()}
        return value

    (root / "designability.json").write_text(json.dumps({
        "arms": {k: str(v) for k, v in arms.items()},
        "num_seq": args.num_seq,
        "temperature": args.temperature,
        "backbone_noise": args.backbone_noise,
        "mpnn_model": args.mpnn_model,
        "mpnn_seed": args.mpnn_seed,
        "fix_motif_sequence": args.fix_motif_sequence,
        "esmfold_model": args.esmfold_model,
        "esmfold_backend": args.esmfold_backend,
        "rmsd_threshold": args.rmsd_threshold,
        "motif_threshold": args.motif_threshold,
        "tm_threshold": args.tm_threshold,
        "motifbench_official": args.motifbench,
        "limit": args.limit,
        "min_length": args.min_length,
        "max_length": args.max_length,
        "summary": finite(summary),
    }, indent=2, allow_nan=False))
    print(f"wrote {root/'designability.json'}", flush=True)


if __name__ == "__main__":
    main()
