#ifndef TET_POISSON_ASSEMBLER_CUH
#define TET_POISSON_ASSEMBLER_CUH

#include "basic/include/options.hpp"

namespace backend {
namespace tet {

const SparseMatrixXr ComputeTetPoissonMatrix(
    const Matrix3Xr& tet_vertices,
    const Matrix4Xi& tet_indices,
    const Options& opt
);


}
}

#endif