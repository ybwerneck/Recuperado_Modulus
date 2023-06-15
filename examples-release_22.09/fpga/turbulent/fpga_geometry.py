from sympy import Symbol, Eq, tanh
import numpy as np
import yaml

from modulus.geometry.primitives_3d import Box, Channel, Plane
from modulus.geometry import Parameterization, Parameter

# check if the geometry is parameterized
with open("./conf/config.yaml", "r") as file:
    yaml_data = yaml.safe_load(file)
parameterized = yaml_data["custom"]["parameterized"]

if parameterized:
    # geometry parameterization
    HS_height = Symbol("HS_height")
    HS_length = Symbol("HS_length")
else:
    HS_length = 0.65
    HS_height = 0.8625

# geometry params for domain
channel_origin = (-2.5, -0.5, -0.5625)
channel_dim = (5.0, 1.0, 1.125)
heat_sink_base_origin = (-0.75, -0.5, -0.4375)
heat_sink_base_dim = (HS_length, 0.05, 0.875)
fin_origin = heat_sink_base_origin
fin_dim = (HS_length, HS_height, 0.0075)
total_fins = 17
flow_box_origin = (-0.85, -0.5, -0.5625)
flow_box_dim = (0.85, 1.0, 1.125)
source_origin = (-0.55, -0.5, -0.125)
source_dim = (0.25, 0.0, 0.25)

# define sympy varaibles to parametize domain curves
x, y, z = Symbol("x"), Symbol("y"), Symbol("z")

# define geometry
# channel
channel = Channel(
    channel_origin,
    (
        channel_origin[0] + channel_dim[0],
        channel_origin[1] + channel_dim[1],
        channel_origin[2] + channel_dim[2],
    ),
)

# fpga heat sink
heat_sink_base = Box(
    heat_sink_base_origin,
    (
        heat_sink_base_origin[0] + heat_sink_base_dim[0],  # base of heat sink
        heat_sink_base_origin[1] + heat_sink_base_dim[1],
        heat_sink_base_origin[2] + heat_sink_base_dim[2],
    ),
)
fin_center = (
    fin_origin[0] + fin_dim[0] / 2,
    fin_origin[1] + fin_dim[1] / 2,
    fin_origin[2] + fin_dim[2] / 2,
)
fin = Box(
    fin_origin,
    (
        fin_origin[0] + fin_dim[0],
        fin_origin[1] + fin_dim[1],
        fin_origin[2] + fin_dim[2],
    ),
)
gap = (heat_sink_base_dim[2] - fin_dim[2]) / (total_fins - 1)  # gap between fins
fin = fin.repeat(
    gap,
    repeat_lower=(0, 0, 0),
    repeat_higher=(0, 0, total_fins - 1),
    center=fin_center,
)
fpga = heat_sink_base + fin

# entire geometry
geo = channel - fpga

# inlet and outlet
inlet = Plane(
    channel_origin,
    (
        channel_origin[0],
        channel_origin[1] + channel_dim[1],
        channel_origin[2] + channel_dim[2],
    ),
    -1,
)
outlet = Plane(
    (channel_origin[0] + channel_dim[0], channel_origin[1], channel_origin[2]),
    (
        channel_origin[0] + channel_dim[0],
        channel_origin[1] + channel_dim[1],
        channel_origin[2] + channel_dim[2],
    ),
    1,
)

# planes for integral continuity
x_pos = Parameter("x_pos")
x_pos_range = {x_pos: (-0.75, 0.0)}
integral_plane = Plane(
    (x_pos, channel_origin[1], channel_origin[2]),
    (x_pos, channel_origin[1] + channel_dim[1], channel_origin[2] + channel_dim[2]),
    1,
    parameterization=Parameterization(x_pos_range),
)
