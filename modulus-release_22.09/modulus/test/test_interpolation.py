from modulus.models.interpolation import interpolation

import torch
import numpy as np


def test_interpolation():
    # set device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # make context grid to do interpolation from
    grid = [(-1, 2, 30), (-1, 2, 30), (-1, 2, 30)]
    np_linspace = [np.linspace(x[0], x[1], x[2]) for x in grid]
    np_mesh_grid = np.meshgrid(*np_linspace, indexing="ij")
    np_mesh_grid = np.stack(np_mesh_grid, axis=0)
    mesh_grid = torch.tensor(np_mesh_grid, dtype=torch.float32).to(device)
    sin_grid = torch.sin(
        mesh_grid[0:1, :, :] + mesh_grid[1:2, :, :] ** 2 + mesh_grid[2:3, :, :] ** 3
    ).to(device)

    # make query points to evaluate on
    nr_points = 100
    query_points = (
        torch.stack(
            [
                torch.linspace(0.0, 1.0, nr_points),
                torch.linspace(0.0, 1.0, nr_points),
                torch.linspace(0.0, 1.0, nr_points),
            ],
            axis=-1,
        )
        .to(device)
        .requires_grad_(True)
    )

    # compute interpolation
    interpolation_types = [
        "nearest_neighbor",
        "linear",
        "smooth_step_1",
        "smooth_step_2",
        "gaussian",
    ]
    for i_type in interpolation_types:
        # perform interpolation
        computed_interpolation = interpolation(
            query_points,
            sin_grid,
            grid=grid,
            interpolation_type=i_type,
            mem_speed_trade=False,
        )

        # compare to numpy
        np_computed_interpolation = computed_interpolation.cpu().detach().numpy()
        np_ground_truth = (
            (
                torch.sin(
                    query_points[:, 0:1]
                    + query_points[:, 1:2] ** 2
                    + query_points[:, 2:3] ** 3
                )
            )
            .cpu()
            .detach()
            .numpy()
        )
        difference = np.linalg.norm(
            (np_computed_interpolation - np_ground_truth) / nr_points
        )

        # verify
        assert difference < 1e-2, "Test failed!"
