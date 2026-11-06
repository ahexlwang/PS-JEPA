'Compute residue representations from isolated motif coordinates.'
from __future__ import annotations

import torch


def motif_tokens(encoder, coords, atom_mask, residue_mask, motif, mode: str = "in_place",
                 radial_reference=None):
    """Per-residue tokens for the motif residues, zero elsewhere, shaped like a full chain."""
    if mode in ("in_place", "masked"):
        return _in_place(encoder, coords, atom_mask, residue_mask, motif, radial_reference)
    if mode == "packed":
        return _packed(encoder, coords, atom_mask, residue_mask, motif)
    raise ValueError(f"unknown motif token mode: {mode!r}")


def _in_place(encoder, coords, atom_mask, residue_mask, motif, radial_reference=None):
    hidden = residue_mask & ~motif
    encoded = _encode(encoder, coords, atom_mask, residue_mask, hidden, radial_reference)
    return encoded * motif.unsqueeze(-1)


def _packed(encoder, coords, atom_mask, residue_mask, motif):
    packed = torch.zeros_like(coords)
    packed_atom = torch.zeros_like(atom_mask)
    packed_residue = torch.zeros_like(residue_mask)
    index_map = []
    for row in range(residue_mask.shape[0]):
        where = motif[row].nonzero().flatten()
        packed[row, : len(where)] = coords[row, where]
        packed_atom[row, : len(where)] = atom_mask[row, where]
        packed_residue[row, : len(where)] = True
        index_map.append(where)
    encoded = _encode(encoder, packed, packed_atom, packed_residue, None)
    out = torch.zeros_like(encoded)
    for row, where in enumerate(index_map):
        out[row, where] = encoded[row, : len(where)]
    return out


def _encode(encoder, coords, atom_mask, residue_mask, hidden, radial_reference=None):
    """The readout encoder's per-residue stream, with `hidden` sites masked out if given.

    `tokenize` does not expose the hidden set, so the readout encoder is called directly. A model
    without one (a wrapped bottleneck encoder, say) has no masked path and may only be used
    unmasked; asking for one is an error rather than a silent fall back to a whole-chain reading,
    which would hand the caller the answer it is trying to approximate.
    """
    if hidden is None:
        with torch.no_grad():
            return encoder.tokenize(coords, atom_mask, residue_mask)[
                "continuous_semantic"].detach().float()
    readout = getattr(encoder, "readout_encoder", None)
    if readout is None:
        raise TypeError(f"{type(encoder).__name__} has no readout encoder to hide residues from")
    with torch.no_grad():
        # Keyword, not positional. The encoder's signature is
        # `(coords, atom_mask, residue_mask, prediction_mask, residue_type, positions,
        # radial_reference)`, so the sixth positional slot is `positions`: passing the
        # reference there handed it to the wrong parameter and left `radial_reference`
        # None, silently giving the crop's own centroid to an encoder trained against the
        # whole chain's.
        _slot, patch, _world = readout()(coords, atom_mask, residue_mask, hidden,
                                         radial_reference=radial_reference)
    return (patch.float() * residue_mask.unsqueeze(-1)).detach()
