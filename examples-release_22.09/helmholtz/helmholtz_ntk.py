from sympy import Symbol, pi, sin

import modulus
from modulus.hydra import to_absolute_path, instantiate_arch, ModulusConfig
from modulus.utils.io import csv_to_dict
from modulus.solver import Solver
from modulus.domain import Domain
from modulus.geometry.primitives_2d import Rectangle
from modulus.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)
from modulus.domain.validator import PointwiseValidator
from modulus.domain.inferencer import PointwiseInferencer
from modulus.key import Key
from modulus.node import Node
from modulus.eq.pdes.wave_equation import HelmholtzEquation
from modulus.utils.io.plotter import ValidatorPlotter


@modulus.main(config_path="conf", config_name="config_ntk")
def run(cfg: ModulusConfig) -> None:

    # make list of nodes to unroll graph on
    wave = HelmholtzEquation(u="u", k=1.0, dim=2)
    wave_net = instantiate_arch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("u")],
        cfg=cfg.arch.fully_connected,
    )
    nodes = wave.make_nodes() + [wave_net.make_node(name="wave_network")]

    # add constraints to solver
    # make geometry
    x, y = Symbol("x"), Symbol("y")
    height = 2
    width = 2
    rec = Rectangle((-width / 2, -height / 2), (width / 2, height / 2))

    # make domain
    domain = Domain()

    # walls
    wall = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"u": 0},
        batch_size=cfg.batch_size.wall,
        lambda_weighting={"u": 1.0},
    )
    domain.add_constraint(wall, "wall")

    # interior
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={
            "helmholtz": -(
                -((pi) ** 2) * sin(pi * x) * sin(4 * pi * y)
                - ((4 * pi) ** 2) * sin(pi * x) * sin(4 * pi * y)
                + 1 * sin(pi * x) * sin(4 * pi * y)
            )
        },
        batch_size=cfg.batch_size.interior,
        bounds={x: (-width / 2, width / 2), y: (-height / 2, height / 2)},
        lambda_weighting={
            "helmholtz": 1.0,
        },
    )
    domain.add_constraint(interior, "interior")

    # validation data
    mapping = {"x": "x", "y": "y", "z": "u"}
    openfoam_var = csv_to_dict(to_absolute_path("validation/helmholtz.csv"), mapping)
    openfoam_invar_numpy = {
        key: value for key, value in openfoam_var.items() if key in ["x", "y"]
    }
    openfoam_outvar_numpy = {
        key: value for key, value in openfoam_var.items() if key in ["u"]
    }

    openfoam_validator = PointwiseValidator(
        nodes=nodes,
        invar=openfoam_invar_numpy,
        true_outvar=openfoam_outvar_numpy,
        batch_size=1024,
        plotter=ValidatorPlotter(),
    )
    domain.add_validator(openfoam_validator)

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()
