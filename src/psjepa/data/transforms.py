from __future__ import annotations

import torch
from scipy.spatial.transform import Rotation


def random_rotation(device: torch.device | None = None, dtype=torch.float32) -> torch.Tensor:
    """Sample a rotation from SO(3) using the SciPy matrix convention."""
    matrix = Rotation.random().as_matrix()
    return torch.as_tensor(matrix, device=device, dtype=dtype)


def center_on_ca(coords: torch.Tensor, atom_mask: torch.Tensor) -> torch.Tensor:
    valid_ca = atom_mask[:, 1]
    if valid_ca.any():
        center = coords[valid_ca, 1].mean(dim=0)
    elif atom_mask.any():
        center = coords[atom_mask].reshape(-1, 3).mean(dim=0)
    else:
        center = coords.new_zeros(3)
    return (coords - center) * atom_mask.unsqueeze(-1)
