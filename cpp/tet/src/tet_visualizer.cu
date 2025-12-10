#include "tet/include/tet_visualizer.cuh"
#include "basic/include/cuda_helper.hpp"
#include "thrust/device_vector.h"
#include "thrust/reduce.h"
#include "thrust/transform.h"
#include "thrust/inner_product.h"
#include "thrust/fill.h"
#include "thrust/copy.h"

namespace backend {
namespace tet {

// CUDA function to compute the cross section of a tetrahedron with XY-plane

__device__ static const real StableIntersection(const real z1, const real z2) {
    const real denominator = z1 - z2;
    if (denominator < 1e-20 && denominator > -1e-20) {
        return 0.5; // arbitrary
    }
    real res = z1 / denominator;
    if (res < 0) res = 0.;
    if (res > 1) res = 1.;
    return res;
}

// A device function that computes one single tetrahedron's intersection with the XY-plane.
__device__ static const IntersectionInfo ComputeTetXYIntersection(const Vector4r& vz) {
    IntersectionInfo info;
    // Type -1: abnormal case: if any z is NaN or Inf, return count = -1.
    if (vz.array().isNaN().any() || vz.array().isInf().any()) {
        info.count = -1;
        return info;
    }
    // Type 0: if all z are positive or all z are negative, no intersection.
    if ((vz.array() > 0).all() || (vz.array() < 0).all()) {
        info.count = 0;
        return info;
    }
    // Type 1: if at least 3 z are zeros, return the three points.
    if ((vz.array() == 0).count() >= 3) {
        info.count = 3;
        integer idx = 0;
        for (integer i = 0; i < 4; ++i) {
            if (vz(i) == 0) {
                info.point_barycentric[idx++] = Vector4r::Unit(i);
                if (idx == 3) return info;
            }
        }
    }
    // Type 2: if one z > 0 and the other three z <= 0, return the three intersection points on the edges.
    if ((vz.array() > 0).count() == 1) {
        info.count = 3;
        integer idx_pos = -1;
        for (integer i = 0; i < 4; ++i) {
            if (vz(i) > 0) {
                idx_pos = i;
                break;
            }
        }
        integer idx = 0;
        for (integer i = 0; i < 4; ++i) {
            if (i == idx_pos) continue;
            const real t = StableIntersection(vz(idx_pos), vz(i));
            Vector4r bary = Vector4r::Zero();
            bary(idx_pos) = 1 - t;
            bary(i) = t;
            info.point_barycentric[idx++] = bary;
        }
        return info;
    }
    // Type 3: if two z > 0 and two z < 0, return the four intersection points on the edges.
    if ((vz.array() > 0).count() == 2 && (vz.array() < 0).count() == 2) {
        info.count = 4;
        integer idx_pos[2];
        integer idx_neg[2];
        integer pos_count = 0;
        integer neg_count = 0;
        for (integer i = 0; i < 4; ++i) {
            if (vz(i) > 0) {
                idx_pos[pos_count++] = i;
            } else {
                idx_neg[neg_count++] = i;
            }
        }
        integer idx = 0;
        for (integer i = 0; i < 2; ++i) {
            for (integer j = 0; j < 2; ++j) {
                const real t = StableIntersection(vz(idx_pos[i]), vz(idx_neg[j]));
                Vector4r bary = Vector4r::Zero();
                bary(idx_pos[i]) = 1 - t;
                bary(idx_neg[j]) = t;
                info.point_barycentric[idx++] = bary;
            }
        }
        return info;
    }
    // The rest case should have one z < 0 and other three z >= 0.
    {
        info.count = 3;
        integer idx_neg = -1;
        for (integer i = 0; i < 4; ++i) {
            if (vz(i) < 0) {
                idx_neg = i;
                break;
            }
        }
        integer idx = 0;
        for (integer i = 0; i < 4; ++i) {
            if (i == idx_neg) continue;
            const real t = StableIntersection(vz(i), vz(idx_neg));
            Vector4r bary = Vector4r::Zero();
            bary(i) = 1 - t;
            bary(idx_neg) = t;
            info.point_barycentric[idx++] = bary;
        }
        return info;
    }

    return info;
}

__device__
static const real Cross2(const Vector2r& v0, const Vector2r& v1) {
    return v0.x() * v1.y() - v0.y() * v1.x();
}

// Line (p0, p1) and line (q0, q1).
__device__
static const bool CrossingEdges(const Vector2r& p0, const Vector2r& p1, const Vector2r& q0, const Vector2r& q1) {
    // Assumptions of this function: (p0, p1) and (q0, q1) are not degenerated to points.
    // Return true iff the set [p0, p1] and [q0, q1] (including their endpoints!) have shared points.
    // This includes corner cases:
    // - Two lines are coaxial, e.g., p0 = (0, 0), p1 = (1, 0), q0 = (-1, 0), q1 = (3, 0).
    // - Two lines are parallel but not coaxial (should return false).
    // Our goal is to resolve all these corner cases correctly.
    //
    // To avoid checking coaxial edges, we first check if the two edges' bounding boxes overlap. If not, we return
    // false. This will correctly return false for coaxial edges that do not cross.
    const Vector2r p_min = p0.cwiseMin(p1);
    const Vector2r p_max = p0.cwiseMax(p1);
    const Vector2r q_min = q0.cwiseMin(q1);
    const Vector2r q_max = q0.cwiseMax(q1);
    const bool no_overlap = (p_max.x() < q_min.x() || q_max.x() < p_min.x()) || (p_max.y() < q_min.y() || q_max.y() < p_min.y());
    if (no_overlap) {
        return false;
    } else {
        const Vector2r p10 = p1 - p0;
        const Vector2r q10 = q1 - q0;
        // Without checking co-axial edges, this condition will return true for co-axial edges that do not cross, e.g.,
        // p0 = (0, 0), p1 = (1, 0), q0 = (2, 0), q1 = (3, 0). This will cause edge-edge distances to be 0 (see the code below)
        // and cause inf in the barrier function.
        return Cross2(q0 - p0, p10) * Cross2(q1 - p0, p10) <= 0 && Cross2(p0 - q0, q10) * Cross2(p1 - q0, q10) <= 0;
    }
}

struct TetXYIntersectionFunctor {
public:
    TetXYIntersectionFunctor(const Vector3r* tet_vertices,
                              const Vector4i* tet_indices,
                              const real* tet_scalar_field,
                              ColoredTriangle* output_triangles)
        : tet_vertices_(tet_vertices),
          tet_indices_(tet_indices),
          tet_scalar_field_(tet_scalar_field),
          output_triangles_(output_triangles) {}

    __device__ void operator()(const integer& tet_idx) const {
        output_triangles_[tet_idx * 2].is_valid = false;
        output_triangles_[tet_idx * 2 + 1].is_valid = false;

        Vector3r tet_verts[4];
        Vector4r vz;
        for (integer i = 0; i < 4; ++i) {
            tet_verts[i] = tet_vertices_[tet_indices_[tet_idx](i)];
            vz(i) = tet_verts[i](2); // z-coordinate
        }
        // Compute intersection.
        IntersectionInfo info = ComputeTetXYIntersection(vz);
        // Case 0: no intersection / abnormal case.
        if (info.count <= 0) return;
        // Case 1: single triangle intersection.
        if (info.count == 3) {
            ColoredTriangle& tri = output_triangles_[tet_idx * 2];
            tri.is_valid = true;
            for (integer i = 0; i < 3; ++i) {
                const Vector4r& bary = info.point_barycentric[i];
                Vector3r pos = Vector3r::Zero();
                real scalar_val = 0;
                for (integer j = 0; j < 4; ++j) {
                    pos += bary(j) * tet_verts[j];
                    scalar_val += bary(j) * tet_scalar_field_[tet_indices_[tet_idx](j)];
                }
                tri.position[i] = pos.head<2>();
                tri.scalar_color(i) = scalar_val;
            }
            return;
        }
        // Case 2: quad intersection -> split into two triangles.
        if (info.count == 4) {
            // First triangle
            {
               ColoredTriangle& tri = output_triangles_[tet_idx * 2];
               tri.is_valid = true;
               for (integer i = 0; i < 3; ++i) {
                   const Vector4r& bary = info.point_barycentric[i];
                   Vector3r pos = Vector3r::Zero();
                   real scalar_val = 0;
                   for (integer j = 0; j < 4; ++j) {
                       pos += bary(j) * tet_verts[j];
                       scalar_val += bary(j) * tet_scalar_field_[tet_indices_[tet_idx](j)];
                   }
                   tri.position[i] = pos.head<2>();
                   tri.scalar_color(i) = scalar_val;
                }
            }
            // Second triangle
            {
                const ColoredTriangle& tri_pre = output_triangles_[tet_idx * 2];
                ColoredTriangle& tri = output_triangles_[tet_idx * 2 + 1];
                tri.is_valid = true;
                // 4-th point
                const Vector4r& bary = info.point_barycentric[3];
                Vector3r pos = Vector3r::Zero();
                real scalar_val = 0;
                for (integer j = 0; j < 4; ++j) {
                    pos += bary(j) * tet_verts[j];
                    scalar_val += bary(j) * tet_scalar_field_[tet_indices_[tet_idx](j)];
                }
                // Check if (p0, p3) have crossing with edge (p1, p2)
                if (CrossingEdges(tri_pre.position[0], pos.head<2>(), tri_pre.position[1], tri_pre.position[2])) {
                    // Use (p1, p2, p3)
                    tri.position[0] = tri_pre.position[1];
                    tri.position[1] = tri_pre.position[2];
                    tri.position[2] = pos.head<2>();
                    tri.scalar_color(0) = tri_pre.scalar_color(1);
                    tri.scalar_color(1) = tri_pre.scalar_color(2);
                    tri.scalar_color(2) = scalar_val;
                    return;
                } else {
                    // Check if (p1, p3) have crossing with edge (p2, p0)
                    if (CrossingEdges(tri_pre.position[1], pos.head<2>(), tri_pre.position[2], tri_pre.position[0])) {
                        // Use (p2, p0, p3)
                        tri.position[0] = tri_pre.position[2];
                        tri.position[1] = tri_pre.position[0];
                        tri.position[2] = pos.head<2>();
                        tri.scalar_color(0) = tri_pre.scalar_color(2);
                        tri.scalar_color(1) = tri_pre.scalar_color(0);
                        tri.scalar_color(2) = scalar_val;
                        return;
                    } else {
                        // Use (p0, p1, p3)
                        tri.position[0] = tri_pre.position[0];
                        tri.position[1] = tri_pre.position[1];
                        tri.position[2] = pos.head<2>();
                        tri.scalar_color(0) = tri_pre.scalar_color(0);
                        tri.scalar_color(1) = tri_pre.scalar_color(1);
                        tri.scalar_color(2) = scalar_val;
                        return;
                    }
                }
            }
        }
    }

    // Input data
    const Vector3r* tet_vertices_; // (num_vertices, 3)
    const Vector4i* tet_indices_;  // (num_tets, 4)
    const real* tet_scalar_field_; // (num_vertices)

    // Output data
    ColoredTriangle* output_triangles_; // (num_tets * 2)
};

// The main function to compute the tetrahedral mesh's intersection with the XY-plane.
// Return <points, colors> of the resulting triangles.
const std::pair<Matrix2Xr, VectorXr> ComputeTetXYIntersectionGpu(
    const MatrixX3r& tet_vertices,
    const MatrixX4i& tet_indices,
    const VectorXr& tet_scalar_field) {
    const integer num_tets = tet_indices.rows();
    // Allocate output buffer.
    thrust::device_vector<ColoredTriangle> d_output_triangles(num_tets * 2);
    // Copy input data to device.
    thrust::device_vector<Vector3r> tet_vertices_dev(tet_vertices.rows());
    std::vector<Vector3r> zero_vecs(tet_vertices.rows(), Vector3r::Zero());
    for (integer i = 0; i < tet_vertices.rows(); ++i) {
        zero_vecs[i](0) = tet_vertices(i, 0);
        zero_vecs[i](1) = tet_vertices(i, 1);
        zero_vecs[i](2) = tet_vertices(i, 2);
    }
    thrust::copy(
        zero_vecs.data(),
        zero_vecs.data() + zero_vecs.size(),
        tet_vertices_dev.begin());
    std::vector<Vector4i> zero_vec4is(tet_indices.rows(), Vector4i::Zero());
    for (integer i = 0; i < tet_indices.rows(); ++i) {
        zero_vec4is[i](0) = tet_indices(i, 0);
        zero_vec4is[i](1) = tet_indices(i, 1);
        zero_vec4is[i](2) = tet_indices(i, 2);
        zero_vec4is[i](3) = tet_indices(i, 3);
    }
    thrust::device_vector<Vector4i> tet_indices_dev(tet_indices.rows());
    thrust::copy(
        zero_vec4is.data(),
        zero_vec4is.data() + zero_vec4is.size(),
        tet_indices_dev.begin());
    thrust::device_vector<real> tet_scalar_field_dev(tet_scalar_field.size());
    thrust::copy(
        tet_scalar_field.data(),
        tet_scalar_field.data() + tet_scalar_field.size(),
        tet_scalar_field_dev.begin());
    // Launch kernel.
    TetXYIntersectionFunctor functor(
        thrust::raw_pointer_cast(tet_vertices_dev.data()),
        thrust::raw_pointer_cast(tet_indices_dev.data()),
        thrust::raw_pointer_cast(tet_scalar_field_dev.data()),
        thrust::raw_pointer_cast(d_output_triangles.data()));
    thrust::counting_iterator<integer> iter_begin(0);
    thrust::counting_iterator<integer> iter_end(num_tets);
    thrust::for_each(
        iter_begin,
        iter_end,
        functor);
    // Copy back results.
    std::vector<ColoredTriangle> h_output_triangles(num_tets * 2);
    thrust::copy(
        d_output_triangles.begin(),
        d_output_triangles.end(),
        h_output_triangles.begin());
    // Collect valid triangles.
    std::vector<Vector2r> triangle_points;
    std::vector<real> triangle_colors;
    for (integer i = 0; i < num_tets * 2; ++i) {
        if (h_output_triangles[i].is_valid) {
            for (integer j = 0; j < 3; ++j) {
                triangle_points.push_back(h_output_triangles[i].position[j]);
                triangle_colors.push_back(h_output_triangles[i].scalar_color(j));
            }
        }
    }
    // Convert to Eigen matrices.
    const integer num_points = static_cast<integer>(triangle_points.size());
    Matrix2Xr points(2, num_points);
    VectorXr colors(num_points);
    for (integer i = 0; i < num_points; ++i) {
        points.col(i) = triangle_points[i];
        colors(i) = triangle_colors[i];
    }
    return std::make_pair(points, colors);
}

}
}