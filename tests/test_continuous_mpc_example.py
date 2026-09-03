from __future__ import annotations

import pytest
import torch

from continuous_mpc_example import _rot6d_to_matrix


def test_rot6d_identity_uses_first_two_columns() -> None:
    rotation = _rot6d_to_matrix(torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]]))

    torch.testing.assert_close(rotation, torch.eye(3).reshape(1, 3, 3))


def test_rot6d_projects_to_a_rotation_matrix() -> None:
    rotation = _rot6d_to_matrix(torch.tensor([[1.0, 0.1, 0.0, 0.2, 1.0, 0.1]]))[0]

    torch.testing.assert_close(rotation.T @ rotation, torch.eye(3), atol=1.0e-6, rtol=0)
    torch.testing.assert_close(torch.linalg.det(rotation), torch.tensor(1.0))


def test_rot6d_rejects_collinear_columns() -> None:
    with pytest.raises(ValueError, match="nonzero and non-collinear"):
        _rot6d_to_matrix(torch.tensor([[1.0, 0.0, 0.0, 2.0, 0.0, 0.0]]))
