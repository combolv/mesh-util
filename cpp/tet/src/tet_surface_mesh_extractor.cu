#include "tet/include/tet_surface_mesh_extractor.cuh"
#include "basic/include/cuda_helper.hpp"
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/unique.h>

namespace backend {
namespace tet {

__device__ __constant__ static integer kTetFaces[4][3] = {
    {1, 2, 3},
    {0, 3, 2},
    {0, 1, 3},
    {0, 2, 1}
};

struct TetToFacesFunctor {
    const Vector3r* vertices;
    const Vector4i* tets;
    OrientedTriangle* out_faces;

    __device__ void operator()(integer tid) {
        const Vector4i tet = tets[tid];

        Vector3r p[4];
        for (integer i = 0; i < 4; ++i)
            p[i] = vertices[tet(i)];

        const Vector3r tet_center =
            (p[0] + p[1] + p[2] + p[3]) * real(0.25);

        for (integer f = 0; f < 4; ++f) {
            OrientedTriangle tri;
            tri.tet_id  = tid;
            tri.face_id = f;
            tri.is_valid = true;

            const integer i0 = kTetFaces[f][0];
            const integer i1 = kTetFaces[f][1];
            const integer i2 = kTetFaces[f][2];

            tri.v[0] = tet(i0);
            tri.v[1] = tet(i1);
            tri.v[2] = tet(i2);

            const Vector3r n =
                (p[i1] - p[i0]).cross(p[i2] - p[i0]);
            const Vector3r fc =
                (p[i0] + p[i1] + p[i2]) * real(1.0 / 3.0);

            // Flip if pointing inward
            if (n.dot(fc - tet_center) < real(0)) {
                integer tmp = tri.v[1];
                tri.v[1] = tri.v[2];
                tri.v[2] = tmp;
            }

            out_faces[tid * 4 + f] = tri;
        }
    }
};


// ============================================================
// Step 2: Duplicate face elimination (internal faces)
// ============================================================

struct FaceKey {
    integer a, b, c;

    __host__ __device__
    FaceKey() : a(0), b(0), c(0) {}   // <-- REQUIRED for Thrust

    __host__ __device__
    FaceKey(integer x, integer y, integer z) {
        a = min(x, min(y, z));
        c = max(x, max(y, z));
        b = x + y + z - a - c;
    }

    __host__ __device__
    bool operator<(const FaceKey& o) const {
        if (a != o.a) return a < o.a;
        if (b != o.b) return b < o.b;
        return c < o.c;
    }

    __host__ __device__
    bool operator==(const FaceKey& o) const {
        return a == o.a && b == o.b && c == o.c;
    }
};

struct MakeFaceKey {
    __host__ __device__
    FaceKey operator()(const OrientedTriangle& t) const {
        return FaceKey(t.v[0], t.v[1], t.v[2]);
    }
};

struct IsSurfaceFace {
    const FaceKey* keys;
    integer n;

    __host__ __device__
    bool operator()(integer i) const {
        if (i > 0 && keys[i] == keys[i - 1]) return false;
        if (i + 1 < n && keys[i] == keys[i + 1]) return false;
        return true;
    }
};


// ============================================================
// Step 3: Extract vertex indices
// ============================================================

struct ExtractVertex {
    integer k;
    __host__ __device__
    integer operator()(const OrientedTriangle& t) const {
        return t.v[k];
    }
};


// ============================================================
// Main entry
// ============================================================

SurfaceMeshResult ComputeSurfaceMesh(
    const Matrix3Xr& tet_vertices,
    const Matrix4Xi& tet_indices) {

    const integer num_tets = tet_indices.cols();

    // --------------------------------------------------------
    // Copy input to device
    // --------------------------------------------------------

    thrust::device_vector<Vector3r> d_vertices(tet_vertices.cols());
    thrust::device_vector<Vector4i> d_tets(num_tets);

    for (integer i = 0; i < tet_vertices.cols(); ++i)
        d_vertices[i] = tet_vertices.col(i);

    for (integer i = 0; i < num_tets; ++i)
        d_tets[i] = tet_indices.col(i);

    // --------------------------------------------------------
    // Step 1: Tet -> faces
    // --------------------------------------------------------

    thrust::device_vector<OrientedTriangle> d_faces(num_tets * 4);

    TetToFacesFunctor face_functor {
        thrust::raw_pointer_cast(d_vertices.data()),
        thrust::raw_pointer_cast(d_tets.data()),
        thrust::raw_pointer_cast(d_faces.data())
    };

    thrust::for_each(
        thrust::counting_iterator<integer>(0),
        thrust::counting_iterator<integer>(num_tets),
        face_functor);

    // --------------------------------------------------------
    // Step 2: Remove internal faces
    // --------------------------------------------------------

    thrust::device_vector<FaceKey> d_keys(d_faces.size());
    thrust::transform(
        d_faces.begin(), d_faces.end(),
        d_keys.begin(), MakeFaceKey());

    thrust::sort_by_key(
        d_keys.begin(), d_keys.end(),
        d_faces.begin());

    thrust::device_vector<integer> d_is_surface(d_faces.size());
    thrust::transform(
        thrust::counting_iterator<integer>(0),
        thrust::counting_iterator<integer>(d_faces.size()),
        d_is_surface.begin(),
        IsSurfaceFace{
            thrust::raw_pointer_cast(d_keys.data()),
            static_cast<integer>(d_faces.size())
        });

    thrust::device_vector<OrientedTriangle> d_surface_faces(d_faces.size());
    auto end_it = thrust::copy_if(
        d_faces.begin(), d_faces.end(),
        d_is_surface.begin(),
        d_surface_faces.begin(),
        thrust::identity<integer>());

    const integer num_faces =
        static_cast<integer>(end_it - d_surface_faces.begin());

    d_surface_faces.resize(num_faces);

    // --------------------------------------------------------
    // Step 3: Renumber vertices
    // --------------------------------------------------------

    thrust::device_vector<integer> d_used_vertices(num_faces * 3);

    for (integer k = 0; k < 3; ++k) {
        thrust::transform(
            d_surface_faces.begin(),
            d_surface_faces.end(),
            d_used_vertices.begin() + k * num_faces,
            ExtractVertex{k});
    }

    thrust::sort(d_used_vertices.begin(), d_used_vertices.end());
    auto v_end = thrust::unique(d_used_vertices.begin(), d_used_vertices.end());
    d_used_vertices.resize(v_end - d_used_vertices.begin());

    // --------------------------------------------------------
    // Copy back + build output
    // --------------------------------------------------------

    std::vector<integer> h_used_vertices(d_used_vertices.size());
    thrust::copy(
        d_used_vertices.begin(), d_used_vertices.end(),
        h_used_vertices.begin());

    std::unordered_map<integer, integer> remap;
    for (integer i = 0; i < (integer)h_used_vertices.size(); ++i)
        remap[h_used_vertices[i]] = i;

    std::vector<OrientedTriangle> h_faces(num_faces);
    thrust::copy(
        d_surface_faces.begin(), d_surface_faces.end(),
        h_faces.begin());

    SurfaceMeshResult result;

    // vertices
    result.vertices.resize(3, h_used_vertices.size());
    for (integer i = 0; i < (integer)h_used_vertices.size(); ++i)
        result.vertices.col(i) = tet_vertices.col(h_used_vertices[i]);

    // faces
    result.faces.resize(3, num_faces);
    result.old_faces.resize(3, num_faces);
    result.parent_tet.resize(num_faces);

    for (integer i = 0; i < num_faces; ++i) {
        for (integer j = 0; j < 3; ++j) {
            const integer old_v = h_faces[i].v[j];
            result.old_faces(j, i) = old_v;
            result.faces(j, i)     = remap[old_v];
        }
        result.parent_tet(i) = h_faces[i].tet_id;
    }

    return result;
}

void TetSurfaceMeshExtractor::Compute() {
    // Call the surface mesh computation function.
    SurfaceMeshResult surf_mesh = ComputeSurfaceMesh(vertices_, elements_);
    Vout_ = surf_mesh.vertices;
    Fout_ = surf_mesh.faces;
    old_faces_ = surf_mesh.old_faces;
    parent_tet_ = surf_mesh.parent_tet;
    computed_ = true;
}

}
}
