import pymeshlab
import numpy as np
import pyvista

# A test tet mesh with 7 points and 2 elements with a shared point.
def create_non_manifold_point_case():
    points = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1]
    ])
    elements = np.array([
        [0, 1, 2, 3],
        [4, 5, 6, 3]
    ])
    return points, elements

points, elements = create_non_manifold_point_case()
# Use pyvista to visualize the non-manifold point case
def visualize_non_manifold_case():
    mesh = pyvista.UnstructuredGrid(
        {pyvista.CellType.TETRA: elements},
        points
    )
    plotter = pyvista.Plotter()
    plotter.add_mesh(mesh, show_edges=True)
    plotter.show()

# Create the surface mesh of the non-manifold case
surf_faces = np.array(
    [
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
        [1, 2, 3],
        [4, 5, 6],
        [4, 5, 3],
        [4, 6, 3],
        [5, 6, 3]
    ]
)
pymesh = pymeshlab.MeshSet()
pymesh.add_mesh(pymeshlab.Mesh(points, surf_faces), "non_manifold_case")
pymesh.generate_splitting_by_connected_components()
print("Number of connected components:", len(pymesh))

visualize_non_manifold_case()