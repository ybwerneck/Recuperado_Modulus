import numpy as np
from sympy import Symbol, sin

import modulus
from modulus.hydra import instantiate_arch, ModulusConfig
from modulus.solver import Solver
from modulus.domain import Domain
from modulus.geometry.primitives_1d import Line1D
from modulus.geometry.parameterization import OrderedParameterization

from modulus.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)

from modulus.domain.validator import PointwiseValidator
from modulus.utils.io import (
    ValidatorPlotter,
)

from modulus.loss.loss import CausalLossNorm

from modulus.key import Key
from modulus.node import Node
from wave_equation import WaveEquation1D


@modulus.main(config_path="conf", config_name="config_causal")
def run(cfg: ModulusConfig) -> None:

    # make list of nodes to unroll graph on
    we = WaveEquation1D(c=1.0)
    wave_net = instantiate_arch(
        input_keys=[Key("x"), Key("t")],
        output_keys=[Key("u")],
        cfg=cfg.arch.fully_connected,
    )
    nodes = we.make_nodes() + [wave_net.make_node(name="wave_network")]

    # add constraints to solver
    # make geometry
    x, t_symbol = Symbol("x"), Symbol("t")
    L = float(np.pi)
    T = 4 * L
    geo = Line1D(0, L, parameterization=OrderedParameterization({t_symbol: (0, T)}, key=t_symbol))
    # make domain
    domain = Domain()

    # initial condition
    IC = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"u": sin(x), "u__t": sin(x)},
        batch_size=cfg.batch_size.IC,
        lambda_weighting={"u": 100.0, "u__t": 1.0},
        parameterization={t_symbol: 0.0},
    )
    domain.add_constraint(IC, "IC")

    # boundary condition
    BC = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"u": 0},
        batch_size=cfg.batch_size.BC,
        lambda_weighting={"u": 100.0},
    )
    domain.add_constraint(BC, "BC")

    # interior
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"wave_equation": 0},
        batch_size=cfg.batch_size.interior,
        loss=CausalLossNorm(eps=1.0),
        fixed_dataset=False,
        shuffle=False
    )
    domain.add_constraint(interior, "interior")

    # add validation data
    deltaT = 0.01
    deltaX = 0.01
    x = np.arange(0, L, deltaX)
    t = np.arange(0, T, deltaT)
    xx, tt = np.meshgrid(x, t)
    X_star = np.expand_dims(xx.flatten(), axis=-1)
    T_star = np.expand_dims(tt.flatten(), axis=-1)
    u = np.sin(X_star) * (np.cos(T_star) + np.sin(T_star))
    invar_numpy = {"x": X_star, "t": T_star}
    outvar_numpy = {"u": u}


    validator = PointwiseValidator(
        nodes=nodes,
        invar=invar_numpy,
        true_outvar=outvar_numpy,
        batch_size=128,
        plotter=ValidatorPlotter()
    )
    domain.add_validator(validator)

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()
