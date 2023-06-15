import numpy as np
from sympy import Symbol, Eq

import modulus
from modulus.hydra import to_absolute_path, instantiate_arch, ModulusConfig
from modulus.utils.io import csv_to_dict
from modulus.solver import Solver
from modulus.domain import Domain
from modulus.geometry import Bounds
from modulus.geometry.primitives_2d import Line, Circle, Channel2D
from modulus.eq.pdes.navier_stokes import NavierStokes
from modulus.eq.pdes.basic import NormalDotVec
from modulus.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)

from modulus.domain.validator import PointwiseValidator
from modulus.key import Key
from modulus import quantity
from modulus.eq.non_dim import NonDimensionalizer, Scaler


@modulus.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:

    # physical quantities
    nu = quantity(0.02, "kg/(m*s)")
    rho = quantity(1.0, "kg/m^3")
    inlet_u = quantity(1.0, "m/s")
    inlet_v = quantity(0.0, "m/s")
    noslip_u = quantity(0.0, "m/s")
    noslip_v = quantity(0.0, "m/s")
    outlet_p = quantity(0.0, "pa")
    velocity_scale = inlet_u
    density_scale = rho
    length_scale = quantity(20, "m")
    nd = NonDimensionalizer(
        length_scale=length_scale,
        time_scale=length_scale / velocity_scale,
        mass_scale=density_scale * (length_scale ** 3),
    )

    # geometry
    channel_length = (quantity(-10, "m"), quantity(30, "m"))
    channel_width = (quantity(-10, "m"), quantity(10, "m"))
    cylinder_center = (quantity(0, "m"), quantity(0, "m"))
    cylinder_radius = quantity(0.5, "m")
    channel_length_nd = tuple(map(lambda x: nd.ndim(x), channel_length))
    channel_width_nd = tuple(map(lambda x: nd.ndim(x), channel_width))
    cylinder_center_nd = tuple(map(lambda x: nd.ndim(x), cylinder_center))
    cylinder_radius_nd = nd.ndim(cylinder_radius)

    channel = Channel2D(
        (channel_length_nd[0], channel_width_nd[0]),
        (channel_length_nd[1], channel_width_nd[1]),
    )
    inlet = Line(
        (channel_length_nd[0], channel_width_nd[0]),
        (channel_length_nd[0], channel_width_nd[1]),
        normal=1,
    )
    outlet = Line(
        (channel_length_nd[1], channel_width_nd[0]),
        (channel_length_nd[1], channel_width_nd[1]),
        normal=1,
    )
    wall_top = Line(
        (channel_length_nd[1], channel_width_nd[0]),
        (channel_length_nd[1], channel_width_nd[1]),
        normal=1,
    )
    cylinder = Circle(cylinder_center_nd, cylinder_radius_nd)
    volume_geo = channel - cylinder

    # make list of nodes to unroll graph on
    ns = NavierStokes(nu=nd.ndim(nu), rho=nd.ndim(rho), dim=2, time=False)
    normal_dot_vel = NormalDotVec(["u", "v"])
    flow_net = instantiate_arch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("u"), Key("v"), Key("p")],
        cfg=cfg.arch.fully_connected,
    )
    nodes = (
        ns.make_nodes()
        + normal_dot_vel.make_nodes()
        + [flow_net.make_node(name="flow_network")]
        + Scaler(
            ["u", "v", "p"],
            ["u_scaled", "v_scaled", "p_scaled"],
            ["m/s", "m/s", "m^2/s^2"],
            nd,
        ).make_node()
    )

    # make domain
    domain = Domain()
    x, y = Symbol("x"), Symbol("y")

    # inlet
    inlet = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=inlet,
        outvar={"u": nd.ndim(inlet_u), "v": nd.ndim(inlet_v)},
        batch_size=cfg.batch_size.inlet,
    )
    domain.add_constraint(inlet, "inlet")

    # outlet
    outlet = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=outlet,
        outvar={"p": nd.ndim(outlet_p)},
        batch_size=cfg.batch_size.outlet,
    )
    domain.add_constraint(outlet, "outlet")

    # full slip (channel walls)
    walls = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=channel,
        outvar={"u": nd.ndim(inlet_u), "v": nd.ndim(inlet_v)},
        batch_size=cfg.batch_size.walls,
    )
    domain.add_constraint(walls, "walls")

    # no slip
    no_slip = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=cylinder,
        outvar={"u": nd.ndim(noslip_u), "v": nd.ndim(noslip_v)},
        batch_size=cfg.batch_size.no_slip,
    )
    domain.add_constraint(no_slip, "no_slip")

    # interior contraints
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=volume_geo,
        outvar={"continuity": 0, "momentum_x": 0, "momentum_y": 0},
        batch_size=cfg.batch_size.interior,
        bounds=Bounds({x: channel_length_nd, y: channel_width_nd}),
    )
    domain.add_constraint(interior, "interior")

    # Loading validation data from CSV
    mapping = {
        "Points:0": "x",
        "Points:1": "y",
        "U:0": "u_scaled",
        "U:1": "v_scaled",
        "p": "p_scaled",
    }
    openfoam_var = csv_to_dict(
        to_absolute_path("openfoam/cylinder_nu_0.020.csv"), mapping
    )
    openfoam_invar_numpy = {
        key: value / length_scale.magnitude
        for key, value in openfoam_var.items()
        if key in ["x", "y"]
    }
    openfoam_outvar_numpy = {
        key: value
        for key, value in openfoam_var.items()
        if key in ["u_scaled", "v_scaled", "p_scaled"]
    }
    openfoam_validator = PointwiseValidator(
        nodes=nodes, invar=openfoam_invar_numpy, true_outvar=openfoam_outvar_numpy
    )
    domain.add_validator(openfoam_validator)

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()
