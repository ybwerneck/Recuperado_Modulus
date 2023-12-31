import numpy as np
from modulus.geometry.tessellation import Tessellation
from modulus.geometry.primitives_3d import Plane
from modulus.utils.io.vtk import var_to_polyvtk

if __name__ == "__main__":
    # number of points to sample
    nr_points = 100000

    # make tesselated geometry from stl file
    geo = Tessellation.from_stl("./stl_files/tessellated_example.stl")

    # tesselated geometries can be combined with primitives
    cut_plane = Plane((0, -1, -1), (0, 1, 1))
    geo = geo & cut_plane

    # sample geometry for plotting in Paraview
    s = geo.sample_boundary(nr_points=nr_points)
    var_to_polyvtk(s, "tessellated_boundary")
    print("Repeated Surface Area: {:.3f}".format(np.sum(s["area"])))
    s = geo.sample_interior(nr_points=nr_points, compute_sdf_derivatives=True)
    var_to_polyvtk(s, "tessellated_interior")
    print("Repeated Volume: {:.3f}".format(np.sum(s["area"])))
