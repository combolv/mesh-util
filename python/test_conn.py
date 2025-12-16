from backend import TetMeshConnectedComponentExtractor
import numpy as np
from tet_mesh import tet2obj
import pymeshlab
import trimesh

# # Use a simple disconnected mesh:
# verts = np.array(
#     [
#         [0, 0, 0],
#         [1, 0, 0],
#         [0, 1, 0],
#         [0, 0, 1],
#         [2, 2, 0],
#         [2, 0, 2],
#         [0, 2, 2],
#         [2, 2, 2]
#     ], dtype=np.float64
# )
# elems = np.array(
#     [
#         [0, 1, 2, 3],
#         [4, 5, 6, 7]
#     ]
# )

verts = np.load("1314/vert_tetra.npy")
elems = np.load("1314/elems_tetra.npy")
print(verts.shape, elems.shape)
extractor = TetMeshConnectedComponentExtractor(verts.T, elems.T)
extractor.Compute()
verts_result, elems_result = extractor.GetExtractedMesh()
info = extractor.GetReportString()
verts, diag = extractor.GetComponentSizes()
print(info)

print(verts.shape)
print(np.max(elems), np.min(elems))
print(verts_result.shape)
print(np.max(elems_result), np.min(elems_result))

obj_vert, obj_face = tet2obj(verts, elems)
ms = pymeshlab.MeshSet()
ms.add_mesh(pymeshlab.Mesh(obj_vert, obj_face), "mesh")
tri_ms = trimesh.Trimesh(vertices=obj_vert, faces=obj_face)
tri_ms.export("test.obj")

# Split connected components
ms.generate_splitting_by_connected_components()

# If more than one component, record error
if len(ms) > 2:
    print("Unexpected??")