from sympy import Symbol, pi, sin, Number, Eq, And

import modulus
from modulus.hydra import instantiate_arch, ModulusConfig
from modulus.solver import Solver
from modulus.domain import Domain
from modulus.geometry.primitives_3d import Box
from modulus.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)
from modulus.domain.inferencer import PointwiseInferencer
from modulus.utils.io.plotter import InferencerPlotter
from modulus.key import Key
from modulus.eq.pdes.electromagnetic import PEC, SommerfeldBC, MaxwellFreqReal

x, y, z = Symbol("x"), Symbol("y"), Symbol("z")


@modulus.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:

    # params for domain
    length = 2
    height = 2
    width = 2

    eigenmode = [1]
    wave_number = 16.0  # wave_number = freq/c
    waveguide_port = Number(0)
    for k in eigenmode:
        waveguide_port += sin(k * pi * y / length) * sin(k * pi * z / height)

    # define geometry
    rec = Box((0, 0, 0), (width, length, height))
    # make list of nodes to unroll graph on
    hm = MaxwellFreqReal(k=wave_number)
    pec = PEC()
    pml = SommerfeldBC()
    wave_net = instantiate_arch(
        input_keys=[Key("x"), Key("y"), Key("z")],
        output_keys=[Key("ux"), Key("uy"), Key("uz")],
        frequencies=("axis,diagonal", [i / 2.0 for i in range(int(wave_number) + 1)]),
        frequencies_params=(
            "axis,diagonal",
            [i / 2.0 for i in range(int(wave_number) + 1)],
        ),
        cfg=cfg.arch.modified_fourier,
    )
    nodes = (
        hm.make_nodes()
        + pec.make_nodes()
        + pml.make_nodes()
        + [wave_net.make_node(name="wave_network")]
    )

    waveguide_domain = Domain()

    wall_PEC = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"PEC_x": 0.0, "PEC_y": 0.0, "PEC_z": 0.0},
        batch_size=cfg.batch_size.PEC,
        lambda_weighting={"PEC_x": 100.0, "PEC_y": 100.0, "PEC_z": 100.0},
        criteria=And(~Eq(x, 0), ~Eq(x, width)),
    )

    waveguide_domain.add_constraint(wall_PEC, "PEC")

    Waveguide_port = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"uz": waveguide_port},
        batch_size=cfg.batch_size.Waveguide_port,
        lambda_weighting={"uz": 100.0},
        criteria=Eq(x, 0),
    )
    waveguide_domain.add_constraint(Waveguide_port, "Waveguide_port")

    ABC = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={
            "SommerfeldBC_real_x": 0.0,
            "SommerfeldBC_real_y": 0.0,
            "SommerfeldBC_real_z": 0.0,
        },
        batch_size=cfg.batch_size.ABC,
        lambda_weighting={
            "SommerfeldBC_real_x": 10.0,
            "SommerfeldBC_real_y": 10.0,
            "SommerfeldBC_real_z": 10.0,
        },
        criteria=Eq(x, width),
    )
    waveguide_domain.add_constraint(ABC, "ABC")

    Interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={
            "Maxwell_Freq_real_x": 0,
            "Maxwell_Freq_real_y": 0.0,
            "Maxwell_Freq_real_z": 0.0,
        },
        batch_size=cfg.batch_size.Interior,
        bounds={x: (0, width), y: (0, length), z: (0, height)},
        lambda_weighting={
            "Maxwell_Freq_real_x": 1.0 / wave_number ** 2,
            "Maxwell_Freq_real_y": 1.0 / wave_number ** 2,
            "Maxwell_Freq_real_z": 1.0 / wave_number ** 2,
        },
        fixed_dataset=False,
    )
    waveguide_domain.add_constraint(Interior, "Interior")

    # add inferencer data
    interior_points = rec.sample_interior(
        10000, bounds={x: (0, width), y: (0, length), z: (0, height)}
    )
    numpy_inference = PointwiseInferencer(
        nodes=nodes,
        invar=interior_points,
        output_names=["ux", "uy", "uz"],
        plotter=InferencerPlotter(),
        batch_size=2048,
    )
    waveguide_domain.add_inferencer(numpy_inference, "Inf" + str(wave_number).zfill(4))

    # make solver
    slv = Solver(cfg, waveguide_domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()
