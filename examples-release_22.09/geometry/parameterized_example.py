from modulus.geometry.primitives_2d import Rectangle, Circle
from modulus.utils.io.vtk import var_to_polyvtk
from modulus.geometry.parameterization import Parameterization, Parameter

if __name__ == "__main__":
    # make plate with parameterized hole
    # make parameterized primitives
    plate = Rectangle(point_1=(-1, -1), point_2=(1, 1))
    y_pos = Parameter("y_pos")
    parameterization = Parameterization({y_pos: (-1, 1)})
    circle = Circle(center=(0, y_pos), radius=0.3, parameterization=parameterization)
    geo = plate - circle

    # sample geometry over entire parameter range
    s = geo.sample_boundary(nr_points=100000)
    var_to_polyvtk(s, "parameterized_boundary")
    s = geo.sample_interior(nr_points=100000)
    var_to_polyvtk(s, "parameterized_interior")

    # sample specific parameter
    s = geo.sample_boundary(
        nr_points=100000, parameterization=Parameterization({y_pos: 0})
    )
    var_to_polyvtk(s, "y_pos_zero_boundary")
    s = geo.sample_interior(
        nr_points=100000, parameterization=Parameterization({y_pos: 0})
    )
    var_to_polyvtk(s, "y_pos_zero_interior")
