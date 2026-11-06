#!/usr/bin/env python3
"""Tokenise a real PDB chain with the frozen gvp_msg encoder, decode it back with a trained
flow decoder, and write both structures as PDB files you can open in PyMOL / ChimeraX / Mol*.

    python scripts/reconstruct_pdb.py --pdb 3kbr.pdb --chain A --out-dir outputs/recon/3kbr_A

Writes <out-dir>/target.pdb (the input backbone, N/CA/C/O only), <out-dir>/prediction.pdb (the
reconstruction, Kabsch-superposed onto the target so the two overlay), <out-dir>/overlay.pdb
(target as chain A, prediction as chain B, one file) and <out-dir>/metrics.json.

The decoder only ever sees the encoder's invariant per-residue vectors: no coordinates, no
sequence. Residue names are carried over from the input purely for display.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from psjepa.config import load_config
from psjepa.data.transforms import center_on_ca
from psjepa.layers.flow_matching import sample_flow
from psjepa.tokenization.semantic_tokenizer import SemanticTokenizer
from psjepa.training.structure_metrics import structure_metrics
from train_decoder import build_decoder, representation

ATOMS = ("N", "CA", "C", "O")


def read_backbone(path: Path, chain: str | None) -> tuple[np.ndarray, np.ndarray, list[str], list[str], str]:
    """N/CA/C/O of one chain from the first model of a PDB file."""
    residues: dict[tuple[str, int, str], dict[str, np.ndarray]] = {}
    names: dict[tuple[str, int, str], str] = {}
    order: list[tuple[str, int, str]] = []
    chosen = chain
    for line in path.read_text().splitlines():
        if line.startswith("ENDMDL"):
            break
        if not line.startswith("ATOM"):
            continue
        atom = line[12:16].strip()
        if atom not in ATOMS:
            continue
        altloc = line[16]
        if altloc not in (" ", "A", "1"):
            continue
        this_chain = line[21]
        if chosen is None:
            chosen = this_chain
        if this_chain != chosen:
            continue
        key = (this_chain, int(line[22:26]), line[26])
        if key not in residues:
            residues[key] = {}
            names[key] = line[17:20]
            order.append(key)
        residues[key].setdefault(atom, np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])]))
    if not order:
        raise SystemExit(f"no backbone atoms for chain {chain!r} in {path}")
    coords = np.zeros((len(order), 4, 3), dtype=np.float32)
    mask = np.zeros((len(order), 4), dtype=bool)
    for i, key in enumerate(order):
        for j, atom in enumerate(ATOMS):
            if atom in residues[key]:
                coords[i, j] = residues[key][atom]
                mask[i, j] = True
    labels = [f"{k[1]}{k[2].strip()}" for k in order]
    return coords, mask, [names[k] for k in order], labels, chosen


def kabsch(mobile: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotation and translations that superpose `mobile` onto `reference` (proper rotation)."""
    mc, rc = mobile.mean(0), reference.mean(0)
    h = (mobile - mc).T @ (reference - rc)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1.0, 1.0, d]) @ vt
    return r, mc, rc


def write_pdb(path: Path, coords: np.ndarray, mask: np.ndarray, resnames: list[str],
              chain: str, remark: str) -> None:
    lines = [f"REMARK 900 {remark}"]
    serial = 1
    for i, (atoms, present, name) in enumerate(zip(coords, mask, resnames), 1):
        for atom, xyz, ok in zip(ATOMS, atoms, present):
            if not ok:
                continue
            lines.append(f"ATOM  {serial:5d} {atom:^4s} {name:>3s} {chain}{i:4d}    "
                         f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}{1.0:6.2f}{0.0:6.2f}          {atom[0]:>2s}")
            serial += 1
    lines.extend(["TER", "END"])
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb", required=True)
    parser.add_argument("--chain", default=None, help="chain id; default: first chain in the file")
    parser.add_argument("--run", default="outputs/diagnostic/flow_v4/framehead_align_fape_big")
    parser.add_argument("--checkpoint", default="last.pt")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    coords_np, mask_np, resnames, labels, chain = read_backbone(Path(args.pdb), args.chain)
    run = Path(args.run)
    config = load_config(run / "resolved_config.yaml")
    if len(coords_np) > int(config["max_residues"]):
        raise SystemExit(f"{len(coords_np)} residues exceeds the decoder's max_residues {config['max_residues']}")
    coords = center_on_ca(torch.from_numpy(coords_np), torch.from_numpy(mask_np)).unsqueeze(0).to(device)
    atom_mask = torch.from_numpy(mask_np).unsqueeze(0).to(device)
    residue_mask = torch.ones(1, len(coords_np), dtype=torch.bool, device=device)

    tokenizer = SemanticTokenizer.from_checkpoint(config["upstream_checkpoint"], device)
    encoder = tokenizer.model.requires_grad_(False).eval().to(device)
    upstream_config = torch.load(config["upstream_checkpoint"], map_location="cpu", weights_only=False)["config"]
    _kind, decoder = build_decoder(config, upstream_config, device)
    decoder.load_state_dict(torch.load(run / args.checkpoint, map_location="cpu", weights_only=False)["decoder"], strict=True)
    decoder.eval()

    torch.manual_seed(args.seed)
    batch = {"coords": coords, "atom_mask": atom_mask, "residue_mask": residue_mask}
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda" and torch.cuda.is_bf16_supported()):
        condition = representation(encoder, batch, config["representation"])
        generated = sample_flow(decoder, condition, atom_mask, residue_mask,
                                float(config["decoder"]["coordinate_scale"]), args.steps,
                                bool(config.get("predict_x1", False)))
    generated = generated.float()
    metrics = structure_metrics(generated, coords.float(), atom_mask)

    target = coords[0].cpu().numpy()
    pred = generated[0].cpu().numpy()
    valid = mask_np
    r, mc, rc = kabsch(pred[valid], target[valid])
    pred_aligned = (pred - mc) @ r + rc
    rmsd = float(np.sqrt(((pred_aligned[valid] - target[valid]) ** 2).sum(-1).mean()))
    ca = valid[:, 1]
    ca_rmsd = float(np.sqrt(((pred_aligned[ca, 1] - target[ca, 1]) ** 2).sum(-1).mean()))

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    source = f"{Path(args.pdb).name} chain {chain}, {len(coords_np)} residues"
    write_pdb(out / "target.pdb", target, valid, resnames, "A", f"input backbone: {source}")
    write_pdb(out / "prediction.pdb", pred_aligned, valid, resnames, "B",
              f"flow reconstruction from gvp_msg tokens ({run.name}), Kabsch-superposed on target")
    both = (out / "target.pdb").read_text().replace("\nTER\nEND\n", "\nTER\n") + \
           "\n".join(l for l in (out / "prediction.pdb").read_text().splitlines() if l.startswith(("ATOM", "TER", "END"))) + "\n"
    (out / "overlay.pdb").write_text(both)
    summary = {"source": source, "residue_labels_first_last": [labels[0], labels[-1]], "run": run.name,
               "checkpoint": args.checkpoint, "sample_steps": args.steps, "seed": args.seed,
               "backbone_rmsd_A": rmsd, "ca_rmsd_A": ca_rmsd, **{k: float(v) for k, v in metrics.items()}}
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in summary.items()}, ensure_ascii=False))
    print(f"wrote {out/'target.pdb'}, {out/'prediction.pdb'}, {out/'overlay.pdb'}")


if __name__ == "__main__":
    main()
