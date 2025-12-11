#ifndef TET_CROSS_SECTION_CUH
#define TET_CROSS_SECTION_CUH
#include "basic/include/log.hpp"

namespace backend {
namespace tet {

struct IntersectionInfo {
    integer count;
    Vector4r point_barycentric[4];
};

struct ColoredTriangle {
    bool is_valid;
    Vector2r position[3];
    Vector3r scalar_color;
};

const std::pair<Matrix2Xr, VectorXr> ComputeTetXYIntersectionGpu(
    const MatrixX3r& tet_vertices,
    const MatrixX4i& tet_indices,
    const VectorXr& tet_scalar_field);

}
}

#endif