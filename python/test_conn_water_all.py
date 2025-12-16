from tet_mesh import tet2obj
from pathlib import Path
from tqdm import tqdm
import numpy as np
import trimesh
import pymeshlab

# Check all the meshes:
base_path = Path("/cephfs_ssd_dutao/kangbo/amg-data/ncg_processed_meshes/")

# List all meshes
mesh_folders = list(base_path.glob("*/"))
all_errors = []
for mesh_folder in tqdm(mesh_folders):
    vert = np.load(mesh_folder / "vert_tetra.npy")
    elem = np.load(mesh_folder / "elems_tetra.npy")
    vertices, faces = tet2obj(vert, elem)
    # Check if obj is watertight
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    if not mesh.is_watertight:
        print(f"Mesh {mesh_folder} is not watertight!")
        all_errors.append(mesh_folder.name)
        continue
    # Check if only 1 component
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertices, faces))
    ms.generate_splitting_by_connected_components()
    if len(ms) > 2:
        print(f"Mesh {mesh_folder} is not 1-component!")
        all_errors.append(mesh_folder.name)
# Save all errors to a file
with open("tet_connectivity_watertight_errors.txt", "w") as f:
    for error in all_errors:
        f.write(f"{error}\n")
print("Checked all meshes. See tet_connectivity_watertight_errors.txt for details.")