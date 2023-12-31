import glob
import numpy as np

from modulus.geometry.tessellation import Tessellation
from modulus.geometry.discrete_geometry import DiscreteGeometry
from modulus.utils.io.vtk import var_to_polyvtk
from modulus.geometry.parameterization import Parameterization, Parameter

if __name__ == "__main__":
    # make geometry for each bracket
    bracket_files = glob.glob("./bracket_stl/*.stl")
    bracket_files.sort()
    brackets = []
    radius = []
    width = []
    for f in bracket_files:
        # get param values
        radius.append(float(f.split("_")[3]))
        width.append(float(f.split("_")[5][:-4]))

        # make geometry
        brackets.append(Tessellation.from_stl(f))

    # make discretely parameterized geometry
    parameterization = Parameterization(
        {
            Parameter("radius"): np.array(radius)[:, None],
            Parameter("width"): np.array(width)[:, None],
        }
    )
    geo = DiscreteGeometry(brackets, parameterization)

    # sample geometry over entire parameter range
    s = geo.sample_boundary(nr_points=1000000)
    var_to_polyvtk(s, "parameterized_bracket_boundary")
    s = geo.sample_interior(nr_points=1000000)
    var_to_polyvtk(s, "parameterized_bracket_interior")
