"""Cartesian pose input adapters used by the MPC example."""

from __future__ import annotations

import torch
from curobo.types import Pose


def rot6d_to_matrix(rotation_6d: torch.Tensor) -> torch.Tensor:
    """Project Zhou 6D rotations (first two matrix columns) to SO(3)."""

    if rotation_6d.ndim != 2 or rotation_6d.shape[1] != 6:
        raise ValueError("rotation 6D values must have shape [N, 6]")
    if not bool(torch.all(torch.isfinite(rotation_6d)).item()):
        raise ValueError("rotation 6D values must be finite")
    first, second = rotation_6d[:, :3], rotation_6d[:, 3:]
    first_norm = torch.linalg.vector_norm(first, dim=1, keepdim=True)
    x_axis = first / torch.clamp(first_norm, min=1.0e-8)
    second_orthogonal = second - (
        torch.sum(x_axis * second, dim=1, keepdim=True) * x_axis
    )
    second_norm = torch.linalg.vector_norm(second_orthogonal, dim=1, keepdim=True)
    if bool(torch.any(first_norm < 1.0e-6).item()) or bool(
        torch.any(second_norm < 1.0e-6).item()
    ):
        raise ValueError("rotation 6D columns must be nonzero and non-collinear")
    y_axis = second_orthogonal / second_norm
    z_axis = torch.linalg.cross(x_axis, y_axis, dim=1)
    return torch.stack((x_axis, y_axis, z_axis), dim=2)


def relative_rot6d_to_quaternion(
    base_rotation: torch.Tensor, rotation_6d: torch.Tensor
) -> torch.Tensor:
    """Compose tool-local rot6D offsets and return wxyz quaternions."""

    rotation = base_rotation.reshape(1, 3, 3) @ rot6d_to_matrix(rotation_6d)
    matrix = torch.eye(4, device=rotation.device, dtype=rotation.dtype).repeat(
        len(rotation), 1, 1
    )
    matrix[:, :3, :3] = rotation
    quaternion = Pose.from_matrix(matrix).quaternion
    if not isinstance(quaternion, torch.Tensor):
        raise TypeError("Pose conversion did not return a quaternion")
    return quaternion
