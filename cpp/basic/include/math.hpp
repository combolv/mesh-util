#ifndef GRADY_BASIC_MATH
#define GRADY_BASIC_MATH

#include "basic/include/config.hpp"
#include "basic/include/log.hpp"
#include "basic/include/options.hpp"

namespace backend {

const real ToReal(const float value);
const real ToReal(const double value);
const real ToReal(const integer value);
const double ToDouble(const real value);

const real Pi();

// |a - b| <= abs_tol + rel_tol * |b|.
const bool IsClose(const real a, const real b, const real abs_tol, const real rel_tol);

// For all i, |ai - bi| <= abs_tol + rel_tol * |bi|.
const bool IsClose(const VectorXr& a, const VectorXr& b, const real abs_tol, const real rel_tol);

// C++ standards did not define 0^0 ("a domain error may occur").
// We will define 0^0 = 1 using this function.
const real SafePower(const real base, const integer exp);

const real Cross(const Vector2r& a, const Vector2r& b);

const integer GetSign(const real val);

// Polar decomposition and SVD.
void PolarDecomposition(const Matrix2r& F, Matrix2r& R, Matrix2r& S);
void PolarDecomposition(const Matrix3r& F, Matrix3r& R, Matrix3r& S);
const std::pair<Matrix2r, Matrix2r> PolarDecompositionDifferential(const Matrix2r& F, const Matrix2r& R, const Matrix2r& S, const Matrix2r& dF);
const std::pair<Matrix3r, Matrix3r> PolarDecompositionDifferential(const Matrix3r& F, const Matrix3r& R, const Matrix3r& S, const Matrix3r& dF);
const std::pair<Matrix4r, Matrix4r> PolarDecompositionDifferential(const Matrix2r& F, const Matrix2r& R, const Matrix2r& S);
const std::pair<Matrix9r, Matrix9r> PolarDecompositionDifferential(const Matrix3r& F, const Matrix3r& R, const Matrix3r& S);

// F = U * sig * Vt.
void Svd(const Matrix2r& F, Matrix2r& U, Vector2r& sig, Matrix2r& V);
void Svd(const Matrix3r& F, Matrix3r& U, Vector3r& sig, Matrix3r& V);
void SvdDifferential(const Matrix2r& F, const Matrix2r& U, const Vector2r& sig, const Matrix2r& V, const Matrix2r& dF,
    Matrix2r& dU, Vector2r& dsig, Matrix2r& dV);
void SvdDifferential(const Matrix3r& F, const Matrix3r& U, const Vector3r& sig, const Matrix3r& V, const Matrix3r& dF,
    Matrix3r& dU, Vector3r& dsig, Matrix3r& dV);

const Matrix2r DeterminantGradient(const Matrix2r& A);
const Matrix3r DeterminantGradient(const Matrix3r& A);
const Matrix2r DeterminantGradientDifferential(const Matrix2r& A, const Matrix2r& dA);
const Matrix3r DeterminantGradientDifferential(const Matrix3r& A, const Matrix3r& dA);
const Matrix4r DeterminantHessian(const Matrix2r& A);
const Matrix9r DeterminantHessian(const Matrix3r& A);
// A -> H.
// dLdH -> dLdA.
const Matrix2r BackpropagateDeterminantHessian(const Matrix2r& A, const Matrix4r& dLdH);
const Matrix3r BackpropagateDeterminantHessian(const Matrix3r& A, const Matrix9r& dLdH);

// Skew symmetric matrix.
const Matrix3r CrossProductMatrix(const Vector3r& a);

template<typename DataType>
const DataType Clip(const DataType value, const DataType low, const DataType high) {
    if (value < low) return low;
    else if (value > high) return high;
    else return value;
}

// Implement n choose k.
const integer Cnk(const integer n, const integer k);

// Build local frame.
const Matrix3r BuildFrameFromUnitZ(const Vector3r& z);
const Matrix3r BuildFrameFromTangents(const Vector3r& u, const Vector3r& v);

// Comparing two VectorXi --- may be useful for sorting them in std::map.
template<integer dim>
struct VectorDiComparator {
    const bool operator()(const Eigen::Matrix<integer, dim, 1>& a, const Eigen::Matrix<integer, dim, 1>& b) const {
        for (integer i = 0; i < dim; ++i) {
            if (a(i) < b(i)) return true;
            if (a(i) > b(i)) return false;
        }
        return false;
    }
};

struct VectorXiComparator {
    const bool operator()(const VectorXi& a, const VectorXi& b) const {
        if (a.size() < b.size()) return true;
        else if (a.size() > b.size()) return false;
        else {
            const integer num = static_cast<integer>(a.size());
            for (integer i = 0; i < num; ++i) {
                if (a(i) < b(i)) return true;
                if (a(i) > b(i)) return false;
            }
            return false;
        }
    }
};

const VectorXi SortVectorXi(const VectorXi& a);
const bool IsSubset(const VectorXi& part, const VectorXi& full);

const bool InsideConvexPolygon2d(const Matrix2Xr& plane_normals, const VectorXr& plane_offsets, const Vector2r& point);
// Input arguments:
// - plane_normals: the number of columns = the number of faces.
// - plane_offsets: its size = the number of faces.
// - Each face falls on plane_normals.col(i).dot(x) + plane_offsets(i) = 0.
// The normal points outward, i.e., plane_normals.T * x + plane_offsets <= 0 represents the polyhedron.
const bool InsideConvexPolyhedron(const Matrix3Xr& plane_normals, const VectorXr& plane_offsets, const Vector3r& point);
const bool InsideTriangle(const Eigen::Matrix<real, 2, 3>& points, const Vector2r& point);
const bool InsideTetrahedron(const Eigen::Matrix<real, 3, 4>& points,  const Vector3r& point);
const bool InsideGeneralPolyhedron(const Matrix3Xr& points, const Matrix4Xi& tets, const Vector3r& point);
const bool InsideCube(const Vector3r& origin, const real size, const Vector3r& point);

template<integer dim>
const Eigen::Matrix<real, dim, dim> GenerateRandomOrthogonalMatrix();

// Return an orthogonal matrix whose first column is v.
template<integer dim>
const Eigen::Matrix<real, dim, dim> GenerateRandomOrthogonalMatrix(const Eigen::Matrix<real, dim, 1>& v);

const VectorXr NormalizeVector(const VectorXr& v);

// Input argument:
// - vertices: 2D vectors of a line segment or 3D vectors of a polygon, arranged counter-clockwise.
const Vector2r ComputeUnitNormal(const Matrix2Xr& vertices);
const Vector3r ComputeUnitNormal(const Matrix3Xr& vertices);

// SPD projection.
const Matrix2r ProjectToSpd(const Matrix2r& A);
const Matrix3r ProjectToSpd(const Matrix3r& A);
const Matrix4r ProjectToSpd(const Matrix4r& A);
const Matrix6r ProjectToSpd(const Matrix6r& A);
const Matrix8r ProjectToSpd(const Matrix8r& A);
const Matrix9r ProjectToSpd(const Matrix9r& A);
const Matrix18r ProjectToSpd(const Matrix18r& A);

// Eigen to std vector.
const std::vector<real> ToStdVector(const VectorXr& v);
const VectorXr ToEigenVector(const std::vector<real>& v);
const VectorXi ToEigenVector(const std::vector<integer>& v);

// Project to geometry.
const Vector2r ProjectToLine(const Vector2r& position, const Vector2r& v0, const Vector2r& v1);
const Vector2r ProjectToTriangle(const Vector2r& position, const Vector2r& v0, const Vector2r& v1, const Vector2r& v2);
const Vector3r ProjectToTriangle(const Vector3r& position, const Vector3r& v0, const Vector3r& v1, const Vector3r& v2);
const Vector3r ProjectToTetrahedron(const Vector3r& position, const Vector3r& v0, const Vector3r& v1, const Vector3r& v2, const Vector3r& v3);

template<integer dim>
const integer ArgMax(const Eigen::Matrix<real, dim, 1>& v) {
    integer d = -1;
    real max_coeff = -std::numeric_limits<real>::infinity();
    for (integer i = 0; i < dim; ++i)
        if (v(i) > max_coeff) {
            max_coeff = v(i);
            d = i;
        }
    return d;
}

template<integer dim>
const integer ArgMin(const Eigen::Matrix<real, dim, 1>& v) {
    integer d = -1;
    real min_coeff = std::numeric_limits<real>::infinity();
    for (integer i = 0; i < dim; ++i)
        if (v(i) < min_coeff) {
            min_coeff = v(i);
            d = i;
        }
    return d;
}

// Polynomial root finding using Eigen.
const std::vector<real> PolynomialRootFinding(const VectorXr& coeffs, const real lower_bound, const real upper_bound);

// Check gradients and Hessian.
const bool CheckGradientAndHessian(const VectorXr& x0,
    const std::function<const real(const VectorXr&)>& func,
    const std::function<const VectorXr(const VectorXr&)>& func_grad,
    const std::function<const MatrixXr(const VectorXr&)>& func_hess,
    const std::function<const bool(const integer)>& skip_dof,
    const Options& opt);

const bool CheckGradientAndHessian(const VectorXr& x0,
    const std::function<const real(const VectorXr&)>& func,
    const std::function<const VectorXr(const VectorXr&)>& func_grad,
    const std::function<const SparseMatrixXr(const VectorXr&)>& func_hess,
    const std::function<const bool(const integer)>& skip_dof,
    const Options& opt);

}

#endif