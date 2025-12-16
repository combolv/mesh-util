#ifndef TET_SURFACE_MESH_HPP
#define TET_SURFACE_MESH_HPP
#include "basic/include/log.hpp"

namespace backend {
namespace tet {

struct SurfaceMeshResult {
    Matrix3Xr vertices;      // 3 x Ns
    Matrix3Xi faces;         // 3 x Fs (new indices)
    Matrix3Xi old_faces;     // 3 x Fs (old vertex indices)
    VectorXi  parent_tet;    // Fs     (which tet produced this face)
};

struct OrientedTriangle {
    integer v[3];       // original vertex indices
    integer tet_id;     // parent tet
    integer face_id;    // local face id [0..3]
    bool is_valid;
};


class TetSurfaceMeshExtractor {
public:
    TetSurfaceMeshExtractor(const Matrix3Xr& vertices, const Matrix4Xi& elements) :
        vertices_(vertices), elements_(elements) {}
    
    void Compute();

    const std::pair<Matrix3Xr, Matrix3Xi> GetSurfaceMesh() const {
        Assert(computed_, "tet::TetSurfaceMeshExtractor::GetSurfaceMesh", "Compute() must be called before GetSurfaceMesh().");
        return {Vout_, Fout_};
    }

    const Matrix3Xi& GetOldFaces() const {
        Assert(computed_, "tet::TetSurfaceMeshExtractor::GetOldFaces", "Compute() must be called before GetOldFaces().");
        return old_faces_;
    }

    const VectorXi& GetParentTet() const {
        Assert(computed_, "tet::TetSurfaceMeshExtractor::GetParentTet", "Compute() must be called before GetParentTet().");
        return parent_tet_;
    }

private:
    bool computed_ = false;
    const Matrix3Xr vertices_;
    const Matrix4Xi elements_;
    Matrix3Xr Vout_;
    Matrix3Xi Fout_;
    // Additional mappings.
    Matrix3Xi old_faces_;     // 3 x Fs (old vertex indices)
    VectorXi  parent_tet_;    // Fs     (which tet produced this face)
};

}
}

#endif