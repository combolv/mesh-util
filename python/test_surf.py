from backend import TetSurfaceMeshExtractor, SurfaceMeshConnectedComponentExtractor
from tet_mesh import tet2obj
import pymeshlab
import trimesh
import numpy as np
import pyvista as pv

tet_verts = np.load("1314/vert_tetra.npy")
tet_indices = np.load("1314/elems_tetra.npy")

plotter = pv.Plotter()
tet_mesh = pv.UnstructuredGrid(
    {pv.CellType.TETRA: tet_indices},
    tet_verts
)

plotter = pv.Plotter()
plotter.add_mesh(tet_mesh, show_edges=True)
plotter.show()
input("?")
extractor = TetSurfaceMeshExtractor(tet_verts.T, tet_indices.T)
extractor.Compute()
surf_verts, surf_faces = extractor.GetSurfaceMesh()

obj_verts, obj_faces = tet2obj(tet_verts, tet_indices)
# Check if surf and obj are in the same shape
print("Surface Mesh Vertices Shape:", surf_verts.shape)
print("Object Mesh Vertices Shape:", obj_verts.shape)
print("Surface Mesh Faces Shape:", surf_faces.shape)
print("Object Mesh Faces Shape:", obj_faces.shape)
# input("?")

surf_conn_checker = SurfaceMeshConnectedComponentExtractor(surf_verts, surf_faces)
surf_conn_checker.Compute()
is_connected = surf_conn_checker.IsConnected()
print("Is the extracted surface mesh connected?", is_connected)


# Export the obj file from the extractor.
trimesh_mesh = trimesh.Trimesh(vertices=surf_verts.T, faces=surf_faces.T)
# Check if watertight.
print("Is the extracted surface mesh watertight?", trimesh_mesh.is_watertight)

trimesh_mesh.export("extracted_surface_mesh.obj")

# Now we fix it using pymeshlab
ms = pymeshlab.MeshSet()
ms.add_mesh(pymeshlab.Mesh(surf_verts.T, surf_faces.T), "extracted_mesh")
ms.generate_splitting_by_connected_components()
print("Number of connected components after splitting:", len(ms))
if len(ms) > 1:
    print("The extracted surface mesh is not connected after all!")
# Export the largest component
# ms.set_current_mesh(1)
# ms.save_current_mesh("extracted_surface_mesh_largest_component.obj")
# Check watertight
largest_comp_mesh = ms[1]
tri_mesh = trimesh.Trimesh(vertices=largest_comp_mesh.vertex_matrix(), faces=largest_comp_mesh.face_matrix())
print("Is the largest component watertight?", tri_mesh.is_watertight)