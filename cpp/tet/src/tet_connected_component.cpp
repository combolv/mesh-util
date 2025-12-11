#include "tet/include/tet_connected_component.hpp"

namespace backend {
namespace tet {


// Main function: find largest connected component by bounding box diagonal and extract it.
// Inputs:
//   V: 3 x Nv matrix (Eigen::Matrix<real,3,-1>) columns are vertex positions
//   T: 4 x Nt matrix (Eigen::Matrix<integer,4,-1>) columns are tetra indices (assumed 0-based)
// Outputs:
//   Vout, Tout: shapes 3 x Nvk and 4 x Mtk with reindexed vertices (0-based)
// Returns ConnectedComponentExtractResult giving connectivity info.
const ConnectedComponentExtractResult ExtractLargestComponentByAABB(
    const Matrix3Xr& V,
    const Matrix4Xi& T,
    Matrix3Xr& Vout,
    Matrix4Xi& Tout
) {
    const integer Nv = static_cast<integer>(V.cols());
    const integer Nt = static_cast<integer>(T.cols());

    ConnectedComponentExtractResult res;
    if (Nv == 0 || Nt == 0) {
        // empty mesh -> return empty outputs and meaningful result
        Vout.resize(3,0);
        Tout.resize(4,0);
        res.is_connected = (Nt == 0 && Nv <= 1);
        res.num_components = (Nv>0) ? Nv : 0;
        res.largest_component_root = (Nv>0) ? 0 : -1;
        res.largest_bbox_diag = 0.0;
        if (Nv>0) res.comp_sizes = std::vector<integer>(res.num_components, 1);
        return res;
    }

    // 1) Build connectivity using DSU: union vertices inside each tetra
    DSU dsu(Nv);
    for (integer ti = 0; ti < Nt; ++ti) {
        for (integer i = 0; i < 3; ++i) {
            for (integer j = i + 1; j < 4; ++j) {
                dsu.unite(T(i,ti), T(j,ti));
            }
        }
    }

    // 2) Collapse find() and build mapping root -> list of vertices
    std::vector<integer> root_of(Nv);
    for (integer v = 0; v < Nv; ++v) root_of[v] = dsu.find(v);

    // Map root value to component index [0..k-1]
    std::unordered_map<integer,integer> root_to_cid;
    root_to_cid.reserve(Nv);
    std::vector<std::vector<integer>> comps_vertices;
    comps_vertices.reserve(32);

    for (integer v = 0; v < Nv; ++v) {
        integer r = root_of[v];
        auto it = root_to_cid.find(r);
        if (it == root_to_cid.end()) {
            integer cid = (integer)comps_vertices.size();
            root_to_cid[r] = cid;
            comps_vertices.emplace_back();
            comps_vertices.back().push_back(v);
        } else {
            comps_vertices[it->second].push_back(v);
        }
    }

    const integer K = static_cast<integer>(comps_vertices.size());
    res.num_components = K;
    res.is_connected = (K == 1);
    res.comp_sizes.resize(K);
    for (integer i = 0; i < K; ++i) res.comp_sizes[i] = static_cast<integer>(comps_vertices[i].size());

    // 3) For each component compute axis-aligned bounding box diagonal length
    std::vector<real> diag_len(K, 0.0);
    for (integer cid = 0; cid < K; ++cid) {
        const auto &verts = comps_vertices[cid];
        // initialize mins and maxs
        Eigen::Vector3d vmin; vmin.setConstant(std::numeric_limits<real>::infinity());
        Eigen::Vector3d vmax; vmax.setConstant(-std::numeric_limits<real>::infinity());
        for (integer vi : verts) {
            Eigen::Vector3d p = V.col(vi);
            vmin = vmin.cwiseMin(p);
            vmax = vmax.cwiseMax(p);
        }
        diag_len[cid] = (vmax - vmin).norm();
    }

    // 4) find component with maximum diagonal
    integer best_cid = 0;
    real best_diag = diag_len[0];
    for (integer cid = 1; cid < K; ++cid) {
        if (diag_len[cid] > best_diag) {
            best_diag = diag_len[cid];
            best_cid = cid;
        }
    }
    res.largest_bbox_diag = best_diag;

    // Need the root representative of chosen component
    integer chosen_root = -1;
    for (auto &p : root_to_cid) {
        if (p.second == best_cid) {
            chosen_root = p.first;
            break;
        }
    }
    res.largest_component_root = chosen_root;

    // 5) Collect tets belonging to chosen component.
    // A tetra belongs if all its 4 vertices are in that component.
    // To speed up: create a boolean mask for vertices in chosen component.
    std::vector<integer> in_chosen(Nv, 0);
    for (integer v : comps_vertices[best_cid]) in_chosen[v] = 1;

    std::vector<integer> kept_tet_indices;
    kept_tet_indices.reserve(Nt);
    for (integer ti = 0; ti < Nt; ++ti) {
        integer a = T(0,ti), b = T(1,ti), c = T(2,ti), d = T(3,ti);
        if (in_chosen[a] && in_chosen[b] && in_chosen[c] && in_chosen[d]) {
            kept_tet_indices.push_back(ti);
        }
    }

    // 6) Re-index used vertices to 0..Nv2-1 and build outputs.
    // Collect unique used vertices in the chosen tets (should match comps_vertices[best_cid] but
    // this guarantees only vertices used by kept tets are included).
    std::vector<integer> used_flag(Nv, 0);
    for (integer ti : kept_tet_indices) {
        used_flag[T(0,ti)] = 1;
        used_flag[T(1,ti)] = 1;
        used_flag[T(2,ti)] = 1;
        used_flag[T(3,ti)] = 1;
    }

    std::vector<integer> old_to_new(Nv, -1);
    integer new_idx = 0;
    for (integer v = 0; v < Nv; ++v) {
        if (used_flag[v]) old_to_new[v] = new_idx++;
    }
    const integer Nv_out = new_idx;
    const integer Nt_out = static_cast<integer>(kept_tet_indices.size());

    Vout.resize(3, Nv_out);
    Tout.resize(4, Nt_out);

    // Fill Vout
    for (integer v = 0; v < Nv; ++v) {
        integer ni = old_to_new[v];
        if (ni >= 0) Vout.col(ni) = V.col(v);
    }

    // Fill Tout with remapped indices
    for (integer oi = 0; oi < Nt_out; ++oi) {
        integer ti = kept_tet_indices[oi];
        Tout(0,oi) = old_to_new[T(0,ti)];
        Tout(1,oi) = old_to_new[T(1,ti)];
        Tout(2,oi) = old_to_new[T(2,ti)];
        Tout(3,oi) = old_to_new[T(3,ti)];
    }

    return res;
}

}
}