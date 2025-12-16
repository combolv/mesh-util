from backend import TetMeshConnectedComponentExtractor
import numpy as np
from pathlib import Path
from tqdm import tqdm

root_path = Path("/cephfs_ssd_dutao/kangbo/amg-data/ncg_processed_meshes")
output_path = Path("/cephfs_ssd_dutao/kangbo/amg-data/ncg_disconnected_meshes")
output_path.mkdir(exist_ok=True)
all_input_list = list(Path(root_path).glob("*/"))
all_abnormal_list = []
all_largest_dominant_list = []
for mesh_path in tqdm(all_input_list):
    verts = np.load(mesh_path / "vert_tetra.npy")
    elems = np.load(mesh_path / "elems_tetra.npy")
    extractor = TetMeshConnectedComponentExtractor(verts.T, elems.T)
    extractor.Compute()
    is_connected = extractor.IsConnected()
    if is_connected:
        continue
    info = extractor.GetReportString()
    verts_result, elems_result = extractor.GetExtractedMesh()
    mesh_output_path = output_path / mesh_path.name
    mesh_output_path.mkdir(exist_ok=True)
    # Write verts and elems
    np.save(mesh_output_path / "vert_tetra.npy", verts_result.T)
    np.save(mesh_output_path / "elems_tetra.npy", elems_result.T)
    vert_cnt, diag_size = extractor.GetComponentSizes()
    np.save(mesh_output_path / "vert_count.npy", np.array(vert_cnt))
    np.save(mesh_output_path / "diag_size.npy", np.array(diag_size))
    def check_largest_dominant(arr):
        sorted_arr = sorted(arr, reverse=True)
        return sorted_arr[0] > sorted_arr[1] * 2
    if np.argmax(vert_cnt) == np.argmax(diag_size):
        if check_largest_dominant(vert_cnt) and check_largest_dominant(diag_size):
            all_largest_dominant_list.append(mesh_path.name)
            print("Largest dominant!")
            print(vert_cnt, diag_size)
    
    # Write error info
    with open(mesh_output_path / "error_info.txt", "w") as f:
        f.write(info)
    print("Error mesh in ", mesh_path.name)
    print(info)
    all_abnormal_list.append(mesh_path.name)

# Write all_abnormal_list
with open(output_path / "disconnected_info.txt", "w") as f:
    f.writelines("\n".join(all_abnormal_list))
with open(output_path / "largest_dominant_info.txt", "w") as f:
    f.writelines("\n".join(all_largest_dominant_list))