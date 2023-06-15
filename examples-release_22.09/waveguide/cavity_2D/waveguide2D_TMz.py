from sympy import Symbol, pi, sin, Number, Eq
from sympy.logic.boolalg import Or

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
from modulus.utils.io.plotter import ValidatorPlotter, InferencerPlotter
from modulus.key import Key
from modulus.eq.pdes.wave_equation import HelmholtzEquation
from modulus.eq.pdes.navier_stokes import GradNormal

x, y = Symbol("x"), Symbol("y")


@modulus.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:

    # params for domain
    height = 2
    width = 2

    eigenmode = [2]
    wave_number = 32.0  # wave_number = freq/c
    waveguide_port = Number(0)
    for k in eigenmode:
        waveguide_port += sin(k * pi * y / height)

    # define geometry
    rec = Rectangle((0, 0), (width, height))
    # make list of nodes to unroll graph on
    hm = HelmholtzEquation(u="u", k=wave_number, dim=2)
    gn = GradNormal(T="u", dim=2, time=False)
    wave_net = instantiate_arch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("u")],
        frequencies=(
            "axis,diagonal",
            [i / 2.0 for i in range(int(wave_number) * 2 + 1)],
        ),
        frequencies_params=(
            "axis,diagonal",
            [i / 2.0 for i in range(int(wave_number) * 2 + 1)],
        ),
        cfg=cfg.arch.modified_fourier,
    )
    nodes = (
        hm.make_nodes()
        + gn.make_nodes()
        + [wave_net.make_node(name="wave_network")]
    )

    waveguide_domain = Domain()

    PEC = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"u": 0.0},
        batch_size=cfg.batch_size.PEC,
        lambda_weighting={"u": 100.0},
        criteria=Or(Eq(y, 0), Eq(y, height)),
    )

    waveguide_domain.add_constraint(PEC, "PEC")

    Waveguide_port = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"u": waveguide_port},
        batch_size=cfg.batch_size.Waveguide_port,
        lambda_weighting={"u": 100.0},
        criteria=Eq(x, 0),
    )
    waveguide_domain.add_constraint(Waveguide_port, "Waveguide_port")

    ABC = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"normal_gradient_u": 0.0},
        batch_size=cfg.batch_size.ABC,
        lambda_weighting={"normal_gradient_u": 10.0},
        criteria=Eq(x, width),
    )
    waveguide_domain.add_constraint(ABC, "ABC")

    Interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"helmholtz": 0.0},
        batch_size=cfg.batch_size.Interior,
        bounds={x: (0, width), y: (0, height)},
        lambda_weighting={
            "helmholtz": 1.0 / wave_number ** 2,
        },
    )
    waveguide_domain.add_constraint(Interior, "Interior")

    # add validation data
    mapping = {"x": "x", "y": "y", "u": "u"}
    validation_var = csv_to_dict(
        to_absolute_path("../validation/2Dwaveguide_32_2.csv"), mapping
    )
    validation_invar_numpy = {
        key: value for key, value in validation_var.items() if key in ["x", "y"]
    }
    validation_outvar_numpy = {
        key: value for key, value in validation_var.items() if key in ["u"]
    }

    csv_validator = PointwiseValidator(
        nodes=nodes,
        invar=validation_invar_numpy,
        true_outvar=validation_outvar_numpy,
        batch_size=2048,
        plotter=ValidatorPlotter(),
    )
    waveguide_domain.add_validator(csv_validator)

    # add inferencer data
    csv_inference = PointwiseInferencer(
        nodes=nodes,
        invar=validation_invar_numpy,
        output_names=["u"],
        plotter=InferencerPlotter(),
        batch_size=2048,
    )
    waveguide_domain.add_inferencer(csv_inference, "inf_data")

    # make solver
    slv = Solver(cfg, waveguide_domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()
