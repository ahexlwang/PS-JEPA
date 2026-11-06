import torch

from psjepa.data.transforms import random_rotation


def test_rotation_matches_column_vector_matrix_convention():
    rotation = random_rotation(dtype=torch.float64)
    identity = torch.eye(3, dtype=torch.float64)
    assert torch.allclose(rotation.T @ rotation, identity, atol=1e-10)
    assert torch.allclose(torch.linalg.det(rotation), torch.tensor(1.0, dtype=torch.float64), atol=1e-10)

    coords = torch.randn(7, 3, dtype=torch.float64)
    column_vector_result = torch.einsum("ij,li->lj", rotation, coords)
    batched_result = torch.einsum("lid,df->lif", coords[:, None, :], rotation).squeeze(1)
    assert torch.allclose(column_vector_result, batched_result)
