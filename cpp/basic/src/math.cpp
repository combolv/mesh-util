#include "basic/include/math.hpp"
#include "basic/include/log.hpp"

namespace backend {

const real ToReal(const float value) {
    return static_cast<real>(value);
}

const real ToReal(const double value) {
    return static_cast<real>(value);
}

const real ToReal(const integer value) {
    return static_cast<real>(value);
}

const double ToDouble(const real value) {
    return static_cast<double>(value);
}

const real Pi() {
    return ToReal(3.141592653589793238462643383);
}

const bool IsClose(const real a, const real b, const real abs_tol, const real rel_tol) {
    return std::abs(a - b) <= std::abs(b) * rel_tol + abs_tol;
}

const bool IsClose(const VectorXr& a, const VectorXr& b, const real abs_tol, const real rel_tol) {
    Assert(a.size() == b.size(), "basic::math::IsClose", "Sizes of two vectors are different.");
    return ((a - b).cwiseAbs().array() <= b.cwiseAbs().array() * rel_tol + abs_tol).all();
}

const real SafePower(const real base, const integer exp) {
    if (exp == 0) return ToReal(1.0);
    else return ToReal(std::pow(base, exp));
}

const real Cross(const Vector2r& a, const Vector2r& b) {
    return a.x() * b.y() - a.y() * b.x();
}

const integer GetSign(const real val) {
    return static_cast<integer>((val > 0) - (val < 0));
}

void PolarDecomposition(const Matrix2r& F, Matrix2r& R, Matrix2r& S) {
    const real x = F(0, 0) + F(1, 1);
    const real y = F(1, 0) - F(0, 1);
    const real scale = ToReal(1.0) / ToReal(std::sqrt(x * x + y * y));
    if (std::isnan(scale) || std::isinf(scale)) {
        // x and y are very close to 0. F is in the following form:
        // [a,  b]
        // [b, -a]
        // It is already symmetric.
        R = Matrix2r::Identity();
    } else {
        const real c = x * scale;
        const real s = y * scale;
        R(0, 0) = c;
        R(0, 1) = -s;
        R(1, 0) = s;
        R(1, 1) = c;
    }
    S = R.transpose() * F;
}

void PolarDecomposition(const Matrix3r& F, Matrix3r& R, Matrix3r& S) {
    const Eigen::JacobiSVD<Matrix3r> svd(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
    const Matrix3r Sig = svd.singularValues().asDiagonal();
    const Matrix3r U = svd.matrixU();
    const Matrix3r V = svd.matrixV();
    R = U * V.transpose();
    S = V * Sig * V.transpose();
}

const std::pair<Matrix2r, Matrix2r> PolarDecompositionDifferential(const Matrix2r& F, const Matrix2r& R, const Matrix2r& S, const Matrix2r& dF) {
    // set W = R^T dR = [  0    x  ]
    //                  [  -x   0  ]
    //
    // R^T dF - dF^T R = WS + SW
    //
    // WS + SW = [ x(s21 - s12)   x(s11 + s22) ]
    //           [ -x[s11 + s22]  x(s21 - s12) ]
    // ----------------------------------------------------
    const Matrix2r lhs = R.transpose() * dF - dF.transpose() * R;
    const real x = (lhs(0, 1) - lhs(1, 0)) / (2 * S.trace());
    Matrix2r W = Matrix2r::Zero();
    W(0, 1) = x;
    W(1, 0) = -x;
    const Matrix2r dR = R * W;
    // F = RS.
    // dF = dR * S + R * dS.
    const Matrix2r dS = R.transpose() * (dF - dR * S);
    return std::make_pair(dR, dS);
}

const std::pair<Matrix4r, Matrix4r> PolarDecompositionDifferential(const Matrix2r& F, const Matrix2r& R, const Matrix2r& S) {
    // lhs01 = R00 * dF01 + R10 * dF11 - dF00 * R01 - dF10 * R11.
    // lhs10 = R01 * dF00 + R11 * dF10 - dF01 * R00 - dF11 * R10.
    Vector4r lhs01(-R(0, 1), -R(1, 1), R(0, 0), R(1, 0));
    const Vector4r x = lhs01 / S.trace();
    // R * [0,  x] = [-xR01, xR00] = [-R01, -R11, R00, R10]x.
    //     [-x, 0]   [-xR11, xR10].
    const Matrix4r dRdF = lhs01 * x.transpose();
    // F = RS.
    // S(F) = R(F).T * F.
    // dS = dR.T * F + R.T * dF.
    //    = (dRdF * dF).reshape(dim, dim).T * F + R.T * dF.
    Matrix4r dSdF = Matrix4r::Zero();
    for (integer i = 0; i < 2; ++i)
        for (integer j = 0; j < 2; ++j) {
            Matrix2r dF = Matrix2r::Zero();
            dF(i, j) = 1;
            dSdF.col(i + j * 2) += (dRdF.col(i + j * 2).reshaped(2, 2).transpose() * F + R.transpose() * dF).reshaped();
        }
    return std::make_pair(dRdF, dSdF);
}

const std::pair<Matrix3r, Matrix3r> PolarDecompositionDifferential(const Matrix3r& F, const Matrix3r& R, const Matrix3r& S, const Matrix3r& dF) {
    const Matrix3r lhs = R.transpose() * dF - dF.transpose() * R;
    Matrix3r A = Matrix3r::Zero();
    A(0, 0) = S(0, 0) + S(1, 1);
    A(1, 1) = S(0, 0) + S(2, 2);
    A(2, 2) = S(1, 1) + S(2, 2);
    A(0, 1) = A(1, 0) = S(1, 2);
    A(0, 2) = A(2, 0) = -S(0, 2);
    A(1, 2) = A(2, 1) = S(0, 1);
    const Matrix3r A_inv = A.inverse();
    const Vector3r b(lhs(0, 1), lhs(0, 2), lhs(1, 2));
    const Vector3r xyz = A_inv * b;
    const real x = xyz(0), y = xyz(1), z = xyz(2);
    Matrix3r W = Matrix3r::Zero();
    W(0, 0) = W(1, 1) = W(2, 2) = 0;
    W(0, 1) = x; W(0, 2) = y;
    W(1, 0) = -x; W(1, 2) = z;
    W(2, 0) = -y; W(2, 1) = -z;
    const Matrix3r dR = R * W;
    const Matrix3r dS = R.transpose() * (dF - dR * S);
    return std::make_pair(dR, dS);
}

const std::pair<Matrix9r, Matrix9r> PolarDecompositionDifferential(const Matrix3r& F, const Matrix3r& R, const Matrix3r& S) {
    // lhs01 = R.col(0).dot(dF.col(1)) - dF.col(0).dot(R.col(1)).
    Vector9r lhs01, lhs02, lhs12;
    lhs01 << -R.col(1), R.col(0), Vector3r::Zero();
    lhs02 << -R.col(2), Vector3r::Zero(), R.col(0);
    lhs12 << Vector3r::Zero(), -R.col(2), R.col(1);
    Matrix3r A = Matrix3r::Zero();
    A(0, 0) = S(0, 0) + S(1, 1);
    A(1, 1) = S(0, 0) + S(2, 2);
    A(2, 2) = S(1, 1) + S(2, 2);
    A(0, 1) = A(1, 0) = S(1, 2);
    A(0, 2) = A(2, 0) = -S(0, 2);
    A(1, 2) = A(2, 1) = S(0, 1);
    const Matrix3r A_inv = A.inverse();
    Matrix3Xr b(3, 9);
    b.row(0) = lhs01; b.row(1) = lhs02; b.row(2) = lhs12;
    const Matrix3Xr xyz = A_inv * b;
    const Vector9r x = xyz.row(0), y = xyz.row(1), z = xyz.row(2);
    Matrix3r W = Matrix3r::Zero();
    W(0, 0) = W(1, 1) = W(2, 2) = 0;
    // R01 * -x + R02 * -y
    // R11 * -x + R12 * -y
    // R21 * -x + R22 * -y
    // R00 * x + R02 * -z
    // R10 * x + R12 * -z
    // R20 * x + R22 * -z
    // R00 * y + R01 * z
    // R10 * y + R11 * z
    // R20 * y + R21 * z
    const Matrix9r dRdF = lhs01 * x.transpose() + lhs02 * y.transpose() + lhs12 * z.transpose();
    // F = RS.
    // S(F) = R(F).T * F.
    // dS = dR.T * F + R.T * dF.
    //    = (dRdF * dF).reshape(dim, dim).T * F + R.T * dF.
    Matrix9r dSdF = Matrix9r::Zero();
    for (integer i = 0; i < 3; ++i)
        for (integer j = 0; j < 3; ++j) {
            Matrix3r dF = Matrix3r::Zero();
            dF(i, j) = 1;
            dSdF.col(i + j * 3) += (dRdF.col(i + j * 3).reshaped(3, 3).transpose() * F + R.transpose() * dF).reshaped();
        }
    return std::make_pair(dRdF, dSdF);
}

void Svd(const Matrix2r& F, Matrix2r& U, Vector2r& sig, Matrix2r& V) {
    const Eigen::JacobiSVD<Matrix2r> svd(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
    sig = svd.singularValues();
    U = svd.matrixU();
    V = svd.matrixV();
}

void Svd(const Matrix3r& F, Matrix3r& U, Vector3r& sig, Matrix3r& V) {
    const Eigen::JacobiSVD<Matrix3r> svd(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
    sig = svd.singularValues();
    U = svd.matrixU();
    V = svd.matrixV();
}

void SvdDifferential(const Matrix2r& F, const Matrix2r& U, const Vector2r& sig, const Matrix2r& V, const Matrix2r& dF,
    Matrix2r& dU, Vector2r& dsig, Matrix2r& dV) {
    // https://j-towns.github.io/papers/svd-derivative.pdf.
    dsig = (U.transpose() * dF * V).diagonal();
    const real eps = 10 * std::numeric_limits<real>::epsilon();
    // Ensure that sig is sorted.
    Assert(sig(0) >= sig(1), "basic::math::SvdDifferential", "SVD singular values should be sorted.");
    const Matrix2r S = sig.asDiagonal();
    const Matrix2r Ut = U.transpose();
    const Matrix2r dP = Ut * dF * V;
    const Matrix2r dPt = dP.transpose();
    Matrix2r Sij = Matrix2r::Zero();
    for (integer i = 0; i < 2; ++i)
        for (integer j = 0; j < 2; ++j) {
            if (i >= j) continue;
            // i < j now.
            // sig(i) >= sig(j).
            if (sig(i) - sig(j) > eps) {
                Sij(i, j) = ToReal(1) / (sig(j) * sig(j) - sig(i) * sig(i));
                Sij(j, i) = -Sij(i, j);
            } else {
                // PrintWarning("Singular values are too similar. SVD derivatives are undefined.");
            }
        }
    const Matrix2r domega_U = Sij.cwiseProduct(dP * S + S * dPt);
    const Matrix2r domega_V = Sij.cwiseProduct(S * dP + dPt * S);
    dU = U * domega_U;
    dV = V * domega_V;
}

void SvdDifferential(const Matrix3r& F, const Matrix3r& U, const Vector3r& sig, const Matrix3r& V, const Matrix3r& dF,
    Matrix3r& dU, Vector3r& dsig, Matrix3r& dV) {
    dsig = (U.transpose() * dF * V).diagonal();
    const real eps = 10 * std::numeric_limits<real>::epsilon();
    // Ensure that sig is sorted.
    Assert(sig(0) >= sig(1), "basic::math::SvdDifferential", "SVD singular values should be sorted.");
    const Matrix3r S = sig.asDiagonal();
    const Matrix3r Ut = U.transpose();
    const Matrix3r dP = Ut * dF * V;
    const Matrix3r dPt = dP.transpose();
    Matrix3r Sij = Matrix3r::Zero();
    for (integer i = 0; i < 3; ++i)
        for (integer j = 0; j < 3; ++j) {
            if (i >= j) continue;
            // i < j now.
            // sig(i) >= sig(j).
            if (sig(i) - sig(j) > eps) {
                Sij(i, j) = ToReal(1) / (sig(j) * sig(j) - sig(i) * sig(i));
                Sij(j, i) = -Sij(i, j);
            } else {
                // PrintWarning("Singular values are too similar. SVD derivatives are undefined.");
            }
        }
    const Matrix3r domega_U = Sij.cwiseProduct(dP * S + S * dPt);
    const Matrix3r domega_V = Sij.cwiseProduct(S * dP + dPt * S);
    dU = U * domega_U;
    dV = V * domega_V;
}

const Matrix2r DeterminantGradient(const Matrix2r& A) {
    // J = |A|.
    // dJdA = J * A^{-T}.
    // A = [a, b]
    //     [c, d]
    // A^{-1} = [d, -b] / J.
    //          [-c, a]
    // A^{-T} = [d, -c] / J.
    //          [-b, a]
    // dJdA = [d, -c]
    //        [-b, a]
    Matrix2r dJdA;
    dJdA << A(1, 1), -A(1, 0),
            -A(0, 1), A(0, 0);
    return dJdA;
}

const Matrix3r DeterminantGradient(const Matrix3r& A) {
    // dJ/dA = JA^-T
    // A = [ a0 | a1 | a2 ].
    // J = a0.dot(a1 x a2).
    // dJ/dA = [ a1 x a2 | a2 x a0 | a0 x a1 ]
    Matrix3r dJdA;
    dJdA.col(0) = A.col(1).cross(A.col(2));
    dJdA.col(1) = A.col(2).cross(A.col(0));
    dJdA.col(2) = A.col(0).cross(A.col(1));
    return dJdA;
}

const Matrix2r DeterminantGradientDifferential(const Matrix2r& A, const Matrix2r& dA) {
    Matrix2r delta;
    delta << dA(1, 1), -dA(1, 0),
            -dA(0, 1), dA(0, 0);
    return delta;
}

const Matrix3r DeterminantGradientDifferential(const Matrix3r& A, const Matrix3r& dA) {
    Matrix3r delta;
    delta.col(0) = dA.col(1).cross(A.col(2)) + A.col(1).cross(dA.col(2));
    delta.col(1) = dA.col(2).cross(A.col(0)) + A.col(2).cross(dA.col(0));
    delta.col(2) = dA.col(0).cross(A.col(1)) + A.col(0).cross(dA.col(1));
    return delta;
}

const Matrix4r DeterminantHessian(const Matrix2r& A) {
    Matrix4r H = Matrix4r::Zero();
    H(3, 0) = 1;
    H(2, 1) = -1;
    H(1, 2) = -1;
    H(0, 3) = 1;
    return H;
}

const Matrix9r DeterminantHessian(const Matrix3r& A) {
    Matrix9r H = Matrix9r::Zero();
    const Matrix3r A0 = CrossProductMatrix(A.col(0));
    const Matrix3r A1 = CrossProductMatrix(A.col(1));
    const Matrix3r A2 = CrossProductMatrix(A.col(2));
    H.block<3, 3>(0, 3) += -A2;
    H.block<3, 3>(0, 6) += A1;
    H.block<3, 3>(3, 0) += A2;
    H.block<3, 3>(3, 6) += -A0;
    H.block<3, 3>(6, 0) += -A1;
    H.block<3, 3>(6, 3) += A0;
    return H;
}

// A -> H.
// dLdH -> dLdA.
const Matrix2r BackpropagateDeterminantHessian(const Matrix2r& A, const Matrix4r& dLdH) {
    return Matrix2r::Zero();
}

const Matrix3r BackpropagateDeterminantHessian(const Matrix3r& A, const Matrix9r& dLdH) {
    // A0 = [A.col(0)].
    // A1 = [A.col(1)].
    // A2 = [A.col(2)].
    // H = [    -A2    A1]
    //     [A2        -A0]
    //     [-A1  A0      ].
    Matrix3r dLdA = Matrix3r::Zero();
    for (integer i = 0; i < 3; ++i) {
        const Matrix3r ei = CrossProductMatrix(Vector3r::Unit(i));
        // A0.
        dLdA(i, 0) += ei.cwiseProduct(dLdH.block<3, 3>(6, 3) - dLdH.block<3, 3>(3, 6)).sum();
        // A1.
        dLdA(i, 1) += ei.cwiseProduct(dLdH.block<3, 3>(0, 6) - dLdH.block<3, 3>(6, 0)).sum();
        // A2.
        dLdA(i, 2) += ei.cwiseProduct(dLdH.block<3, 3>(3, 0) - dLdH.block<3, 3>(0, 3)).sum();
    }

    return dLdA;
}

const Matrix3r CrossProductMatrix(const Vector3r& a) {
    Matrix3r A = Matrix3r::Zero();
    A(1, 0) = a.z();
    A(2, 0) = -a.y();
    A(0, 1) = -a.z();
    A(2, 1) = a.x();
    A(0, 2) = a.y();
    A(1, 2) = -a.x();
    return A;
}

const integer Cnk(const integer n, const integer k) {
    Assert(n > 0 && 0 <= k && k <= n, "basic::math::Cnk", "Invalid n or k.");
    if (k == 0) return 1;
    const integer k2 = 2 * k < n ? k : (n - k);
    integer cnk = 1;
    for (integer i = 0; i < k2; ++i) {
        cnk = (cnk * (n - i)) / (i + 1);
    }
    return cnk;
}

const Matrix3r BuildFrameFromUnitZ(const Vector3r& z) {
    // We assume z already has unit length.
    real max_len = -1;
    Matrix3r R; R.setZero();
    R.col(2) = z;
    for (integer i = 0; i < 3; ++i) {
        const Vector3r x = z.cross(Vector3r::Unit(i));
        const real x_len = x.norm();
        if (x_len > max_len) {
            max_len = x_len;
            R.col(0) = x / x_len;
        }
    }
    R.col(1) = z.cross(R.col(0));
    return R;
}

const Matrix3r BuildFrameFromTangents(const Vector3r& u, const Vector3r& v) {
    const real u_len = u.norm();
    const real v_len = v.norm();
    const Vector3r w = u.cross(v);
    const real w_len = w.norm();
    if (w_len > u_len * v_len * ToReal(1e-6)) return BuildFrameFromUnitZ(w / w_len);
    else return BuildFrameFromUnitZ(u_len > v_len ? u / u_len : v / v_len);
}

const VectorXi SortVectorXi(const VectorXi& a) {
    std::vector<integer> a_vec(a.data(), a.data() + a.size());
    // Sort.
    std::sort(a_vec.begin(), a_vec.end());
    // See this thread for why Eigen::Unaligned is needed here:
    // https://stackoverflow.com/questions/17036818/initialise-eigenvector-with-stdvector.
    const VectorXi a_sorted = Eigen::Map<VectorXi, Eigen::Unaligned>(a_vec.data(), a_vec.size());
    return a_sorted;
}

const bool IsSubset(const VectorXi& part, const VectorXi& full) {
    const integer part_num = static_cast<integer>(part.size());
    const integer full_num = static_cast<integer>(full.size());
    if (part_num == 0) return true;
    if (full_num == 0) return false;

    for (integer i = 0; i < part_num; ++i) {
        const integer key = part(i);
        bool found = false;
        for (integer j = 0; j < full_num; ++j) {
            if (key == full(j)) {
                found = true;
                break;
            }
        }
        if (!found) return false;
    }
    return true;
}

const bool InsideConvexPolygon2d(const Matrix2Xr& plane_normals, const VectorXr& plane_offsets, const Vector2r& point) {
    Assert(plane_normals.cols() == plane_offsets.size(), "basic::InsideConvexPolygon2d", "Incompatible plane numbers.");
    return ((VectorXr(point.transpose() * plane_normals) + plane_offsets).array() <= 0).all();
}

const bool InsideConvexPolyhedron(const Matrix3Xr& plane_normals, const VectorXr& plane_offsets, const Vector3r& point) {
    Assert(plane_normals.cols() == plane_offsets.size(), "basic::InsideConvexPolyhedron", "Incompatible plane numbers.");
    return ((VectorXr(point.transpose() * plane_normals) + plane_offsets).array() <= 0).all();
}

const bool InsideTriangle(const Eigen::Matrix<real, 2, 3>& points, const Vector2r& point) {
    // v0 + alpha * (v1 - v0) + beta * (v2 - v0) = position.
    Matrix2r A;
    A.col(0) = points.col(1) - points.col(0);
    A.col(1) = points.col(2) - points.col(0);
    const Vector2r b = point - points.col(0);
    const Vector2r x = A.inverse() * b;
    return x.minCoeff() >= 0 && x.maxCoeff() <= 1 && x.sum() <= 1;
}

const bool InsideTetrahedron(const Eigen::Matrix<real, 3, 4>& points, const Vector3r& point) {
    Matrix4r A; Vector4r b;
    A.topRows(3) = points;
    A.row(3) = Vector4r::Ones();
    b.head(3) = point;
    b(3) = 1;
    const Vector4r x = A.inverse() * b;
    return x.minCoeff() >= 0;
}

const bool InsideGeneralPolyhedron(const Matrix3Xr& points, const Matrix4Xi& tets, const Vector3r& point) {
    const integer tet_num = static_cast<integer>(tets.cols());
    for (integer i = 0; i < tet_num; ++i) {
        Matrix3Xr tet = MatrixXr::Zero(3, 4);
        for (integer j = 0; j < 4; ++j)
            tet.col(j) = points.col(tets(j, i));
        if (InsideTetrahedron(tet, point))
            return true;
    }
    return false;
}

const bool InsideCube(const Vector3r& origin, const real size, const Vector3r& point) {
    return (point - origin).minCoeff() >= 0 && (point - origin).maxCoeff() <= size;
}

template<integer dim>
const Eigen::Matrix<real, dim, dim> GenerateRandomOrthogonalMatrix() {
    const Eigen::Matrix<real, dim, dim> Q = Eigen::Matrix<real, dim, dim>::Random().colPivHouseholderQr().householderQ();
    Assert((Q.transpose() * Q - Eigen::Matrix<real, dim, dim>::Identity()).norm() < dim * ToReal(1e-3),
        "basic::GenerateRandomOrthogonalMatrix", "Invalid rotation matrix.");
    return Q;
}

template
const Matrix2r GenerateRandomOrthogonalMatrix();
template
const Matrix3r GenerateRandomOrthogonalMatrix();
template
const Matrix4r GenerateRandomOrthogonalMatrix();

template<integer dim>
const Eigen::Matrix<real, dim, dim> GenerateRandomOrthogonalMatrix(const Eigen::Matrix<real, dim, 1>& v) {
    Eigen::Matrix<real, dim, dim> R; R.setZero();
    R.col(0) = NormalizeVector(v);
    for (integer j = 1; j < dim; ++j) {
        Eigen::Matrix<real, dim, 1> u = Eigen::Matrix<real, dim, 1>::Random();
        for (integer k = 0; k < j; ++k) {
            u -= u.dot(R.col(k)) * R.col(k);
        }
        R.col(j) = NormalizeVector(u);
    }
    return R;
}

template
const Matrix2r GenerateRandomOrthogonalMatrix(const Vector2r& v);
template
const Matrix3r GenerateRandomOrthogonalMatrix(const Vector3r& v);
template
const Matrix4r GenerateRandomOrthogonalMatrix(const Vector4r& v);

const VectorXr NormalizeVector(const VectorXr& v) {
    const VectorXr unit_v = v.normalized();
    // It is OK to use a hardcoded value 1e-3 because unit_v is either close to 1 or close to 0 (a copy of itself).
    Assert(IsClose(unit_v.norm(), 1, 1e-3, 0), "basic::GenerateRandomOrthogonalMatrix", "Near-zero vector.");
    return unit_v;
}

const Vector2r ComputeUnitNormal(const Matrix2Xr& vertices) {
    const integer vertex_num = static_cast<integer>(vertices.cols());
    Assert(vertex_num == 2, "basic::ComputeUnitNormal", "A line segment should only have 2 vertices.");
    const Vector2r v0 = vertices.col(0);
    const Vector2r v1 = vertices.col(1);
    const Vector2r d = v1 - v0;
    const Vector2r n = Vector2r(d(1), -d(0));
    return NormalizeVector(n);
}

const Vector3r ComputeUnitNormal(const Matrix3Xr& vertices) {
    Assert(static_cast<integer>(vertices.cols()) >= 3, "basic::ComputeUnitNormal", "A polygon should contain at least 3 vertices.");
    const Vector3r average = vertices.rowwise().mean();
    const Vector3r v0 = vertices.col(0);
    const Vector3r v1 = vertices.col(1);
    const Vector3r n = (v0 - average).cross(v1 - average);
    return NormalizeVector(n);
}

const Matrix2r ProjectToSpd(const Matrix2r& A) {
    Eigen::SelfAdjointEigenSolver<Matrix2r> eig_solver(A);
    const Vector2r& la = eig_solver.eigenvalues();
    const Matrix2r& V = eig_solver.eigenvectors();
    return V * la.cwiseMax(Vector2r::Zero()).asDiagonal() * V.transpose();
}

const Matrix3r ProjectToSpd(const Matrix3r& A) {
    Eigen::SelfAdjointEigenSolver<Matrix3r> eig_solver(A);
    const Vector3r& la = eig_solver.eigenvalues();
    const Matrix3r& V = eig_solver.eigenvectors();
    return V * la.cwiseMax(Vector3r::Zero()).asDiagonal() * V.transpose();
}

const Matrix4r ProjectToSpd(const Matrix4r& A) {
    Eigen::SelfAdjointEigenSolver<Matrix4r> eig_solver(A);
    const Vector4r& la = eig_solver.eigenvalues();
    const Matrix4r& V = eig_solver.eigenvectors();
    return V * la.cwiseMax(Vector4r::Zero()).asDiagonal() * V.transpose();
}

const Matrix6r ProjectToSpd(const Matrix6r& A) {
    Eigen::SelfAdjointEigenSolver<Matrix6r> eig_solver(A);
    const Vector6r& la = eig_solver.eigenvalues();
    const Matrix6r& V = eig_solver.eigenvectors();
    return V * la.cwiseMax(Vector6r::Zero()).asDiagonal() * V.transpose();
}

const Matrix8r ProjectToSpd(const Matrix8r& A) {
    Eigen::SelfAdjointEigenSolver<Matrix8r> eig_solver(A);
    const Vector8r& la = eig_solver.eigenvalues();
    const Matrix8r& V = eig_solver.eigenvectors();
    return V * la.cwiseMax(Vector8r::Zero()).asDiagonal() * V.transpose();
}

const Matrix9r ProjectToSpd(const Matrix9r& A) {
    Eigen::SelfAdjointEigenSolver<Matrix9r> eig_solver(A);
    const Vector9r& la = eig_solver.eigenvalues();
    const Matrix9r& V = eig_solver.eigenvectors();
    return V * la.cwiseMax(Vector9r::Zero()).asDiagonal() * V.transpose();
}

const Matrix18r ProjectToSpd(const Matrix18r& A) {
    Eigen::SelfAdjointEigenSolver<Matrix18r> eig_solver(A);
    const Vector18r& la = eig_solver.eigenvalues();
    const Matrix18r& V = eig_solver.eigenvectors();
    return V * la.cwiseMax(Vector18r::Zero()).asDiagonal() * V.transpose();
}

const std::vector<real> ToStdVector(const VectorXr& v) {
    const integer n = static_cast<integer>(v.size());
    std::vector<real> v_vec(n, 0);
    for (integer i = 0; i < n; ++i) v_vec[i] = v(i);
    return v_vec;
}

const VectorXr ToEigenVector(const std::vector<real>& v) {
    const integer n = static_cast<integer>(v.size());
    VectorXr v_eig = VectorXr::Zero(n);
    for (integer i = 0; i < n; ++i) v_eig(i) = v[i];
    return v_eig;
}

const VectorXi ToEigenVector(const std::vector<integer>& v) {
    const integer n = static_cast<integer>(v.size());
    VectorXi v_eig = VectorXi::Zero(n);
    for (integer i = 0; i < n; ++i) v_eig(i) = v[i];
    return v_eig;
}

const Vector2r ProjectToLine(const Vector2r& position, const Vector2r& v0, const Vector2r& v1) {
    // <v0 + t * (v1 - v0) - p, v1 - v0> = 0.
    real t = -(v0 - position).dot(v1 - v0) / (v1 - v0).squaredNorm();
    t = Clip<real>(t, 0, 1);
    return v0 + t * (v1 - v0);
}

const Vector2r ProjectToTriangle(const Vector2r& position, const Vector2r& v0, const Vector2r& v1, const Vector2r& v2) {
    // v0 + alpha * (v1 - v0) + beta * (v2 - v0) = position.
    Matrix2r A;
    A.col(0) = v1 - v0;
    A.col(1) = v2 - v0;
    const Vector2r b = position - v0;
    const Vector2r x = A.inverse() * b;
    if (x.minCoeff() >= 0 && x.maxCoeff() <= 1 && x.sum() <= 1) return position;

    // The projection lies on the edge.
    const Vector2r p0 = ProjectToLine(position, v0, v1);
    const Vector2r p1 = ProjectToLine(position, v1, v2);
    const Vector2r p2 = ProjectToLine(position, v2, v0);
    const real d0 = (p0 - position).squaredNorm();
    const real d1 = (p1 - position).squaredNorm();
    const real d2 = (p2 - position).squaredNorm();
    if (d0 <= d1 && d0 <= d2) return p0;
    else if (d1 <= d0 && d1 <= d2) return p1;
    else return p2;
}

const Vector3r ProjectToTriangle(const Vector3r& position, const Vector3r& v0, const Vector3r& v1, const Vector3r& v2) {
    const Vector3r p1 = v1 - v0;
    const Vector3r p2 = v2 - v0;
    const Vector3r z = p1.cross(p2).normalized();
    const Vector3r x = p1.normalized();
    const Vector3r y = z.cross(x);
    Matrix3r Rt;
    Rt.row(0) = x;
    Rt.row(1) = y;
    Rt.row(2) = z;

    const Vector2r pos_2d = (Rt * (position - v0)).head(2);
    const Vector2r v0_2d(0, 0);
    const Vector2r v1_2d = (Rt * (v1 - v0)).head(2);
    const Vector2r v2_2d = (Rt * (v2 - v0)).head(2);
    const Vector2r proj_2d = ProjectToTriangle(pos_2d, v0_2d, v1_2d, v2_2d);
    return v0 + proj_2d.x() * x + proj_2d.y() * y;
}

const Vector3r ProjectToTetrahedron(const Vector3r& position, const Vector3r& v0, const Vector3r& v1, const Vector3r& v2, const Vector3r& v3) {
    // v0 + alpha * (v1 - v0) + beta * (v2 - v0) + gamma * (v3 - v0) = position.
    Matrix3r A;
    A.col(0) = v1 - v0;
    A.col(1) = v2 - v0;
    A.col(2) = v3 - v0;
    const Vector3r b = position - v0;
    const Vector3r x = A.inverse() * b;
    if (x.minCoeff() >= 0 && x.maxCoeff() <= 1 && x.sum() <= 1) return position;

    // The projection lies on the surface.
    const Vector3r p0 = ProjectToTriangle(position, v0, v1, v2);
    const Vector3r p1 = ProjectToTriangle(position, v1, v2, v3);
    const Vector3r p2 = ProjectToTriangle(position, v2, v3, v0);
    const Vector3r p3 = ProjectToTriangle(position, v3, v0, v1);
    const real d0 = (p0 - position).squaredNorm();
    const real d1 = (p1 - position).squaredNorm();
    const real d2 = (p2 - position).squaredNorm();
    const real d3 = (p3 - position).squaredNorm();

    if (d0 <= d1 && d0 <= d2 && d0 <= d3) return p0;
    else if (d1 <= d0 && d1 <= d2 && d1 <= d3) return p1;
    else if (d2 <= d0 && d2 <= d1 && d2 <= d3) return p2;
    else return p3;
}

const std::vector<real> PolynomialRootFinding(const VectorXr& coeffs, const real lower_bound, const real upper_bound) {
    const integer deg = static_cast<integer>(coeffs.size()) - 1;
    Assert(deg >= 1 && std::abs(coeffs(deg)) != 0, "basic::math::PolynomialRootFinding", "Degenerated polynomials.");
    // Trivial case.
    const std::vector<real> empty;
    if (deg == 1) {
        const real root = -coeffs(0) / coeffs(1);
        if (root <= lower_bound || root > upper_bound) return empty;
        else return std::vector<real>(1, root);
    } else if (deg == 2) {
        // Quadratic.
        const real a = coeffs(2);
        const real b = coeffs(1);
        const real c = coeffs(0);
        const real delta = b * b - 4 * a * c;
        if (delta < 0) return empty;
        if (delta == 0) {
            const real root = -b / (2 * a);
            if (root <= lower_bound || root > upper_bound) return empty;
            else return std::vector<real>(1, root);
        }
        const real sqrt_delta = std::sqrt(delta);
        // x1 = (-b + sqrt_delta) / 2a;
        // x2 = (-b - sqrt_delta) / 2a = 2c / (-b + sqrt_deta);
        const integer sign_b = (b > 0) ? 1 : -1;
        const real neg_b_sqrt_delta = -(b + sign_b * sqrt_delta);
        real x1 = (2 * c) / neg_b_sqrt_delta;
        real x2 = neg_b_sqrt_delta / (2 * a);
        if (x1 > x2) {
            const real swap = x1; x1 = x2; x2 = swap;
        }
        // Now x1 <= x2.
        std::vector<real> roots;
        if (lower_bound < x1 && x1 <= upper_bound) roots.push_back(x1);
        if (lower_bound < x2 && x2 <= upper_bound) roots.push_back(x2);
        return roots;
    } else {
        // Cubic and higher-degree poly.
        Eigen::PolynomialSolver<real, Eigen::Dynamic> solver;
        solver.compute(coeffs);
        std::vector<real> all_real_roots;
        solver.realRoots(all_real_roots);

        std::vector<real> result;
        for (const real r : all_real_roots) {
            if (lower_bound < r && r <= upper_bound) result.push_back(r);
        }
        std::sort(result.begin(), result.end());
        return result;
    }
}

const bool CheckGradientAndHessian(const VectorXr& x0,
    const std::function<const real(const VectorXr&)>& func,
    const std::function<const VectorXr(const VectorXr&)>& func_grad,
    const std::function<const MatrixXr(const VectorXr&)>& func_hess,
    const std::function<const bool(const integer)>& skip_dof,
    const Options& opt) {

    const std::string error_location = "basic::CheckGradientAndHessian";

    const integer verbose = opt.integer_option().HasKey("verbose") ? opt.integer_option()["verbose"] : 0;

    Assert(opt.real_option().HasKey("grad_check_abs_tol") && opt.real_option().HasKey("grad_check_rel_tol")
        && opt.real_option().HasKey("hess_check_abs_tol") && opt.real_option().HasKey("hess_check_rel_tol"),
        error_location, "Missing keys: grad_check_abs_tol, grad_check_rel_tol, hess_check_abs_tol, or hess_check_rel_tol.");
    const real grad_abs_tol = opt.real_option()["grad_check_abs_tol"];
    const real grad_rel_tol = opt.real_option()["grad_check_rel_tol"];
    const real hess_abs_tol = opt.real_option()["hess_check_abs_tol"];
    const real hess_rel_tol = opt.real_option()["hess_check_rel_tol"];

    const integer n = static_cast<integer>(x0.size());
    Assert(n > 0, error_location, "Empty x0.");

    const VectorXr g0 = func_grad(x0);
    Assert(g0.size() == x0.size(), error_location, "Dimension of g and x do not agree.");
    const MatrixXr H0 = func_hess(x0);
    Assert(H0.rows() == H0.cols() && H0.cols() == x0.size(), error_location, "Dimension of H and x do not agree.");

    // Check gradients.
    for (integer i = 0; i < n; ++i) {
        if (skip_dof(i)) continue;

        bool success = false;
        // Try a series of eps.
        for (const real eps : { 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12 }) {
            VectorXr x_pos = x0;
            x_pos(i) += eps;
            VectorXr x_neg = x0;
            x_neg(i) -= eps;
            real E_pos, E_neg;
            try {
                E_pos = func(x_pos);
                E_neg = func(x_neg);
            } catch (std::runtime_error& ) {
                continue;
            }
            const real grad = (E_pos - E_neg) / (2 * eps);
            if (verbose > 0) {
                std::cout << "Grad at " << i << " (eps = " << eps << "): "
                    << g0(i) << " (" << std::abs(g0(i) - grad) << ")" << std::endl;
            }
            if (IsClose(grad, g0(i), grad_abs_tol, grad_rel_tol)) {
                success = true;
                break;
            }
        }
        Assert(success, error_location, "Grad check failed.");
        if (!success) return false;
    }

    // Check the Hessian.
    VectorXr mask = VectorXr::Ones(n);
    for (integer i = 0; i < n; ++i)
        if (skip_dof(i)) mask(i) = 0;

    for (integer i = 0; i < n; ++i) {
        if (skip_dof(i)) continue;

        bool success = false;
        // Try a series of eps.
        for (const real eps : { 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12 }) {
            VectorXr x_pos = x0;
            x_pos(i) += eps;
            VectorXr x_neg = x0;
            x_neg(i) -= eps;
            VectorXr g_pos, g_neg;
            try {
                g_pos = func_grad(x_pos);
                g_neg = func_grad(x_neg);
            } catch (std::runtime_error& ) {
                continue;
            }
            // Check one column of the Hessian matrix.
            const VectorXr h_numerical = ((g_pos - g_neg) / (2 * eps)).cwiseProduct(mask);
            const VectorXr h_analytical = H0.col(i).cwiseProduct(mask);
            if (verbose > 0) {
                std::cout << "Hessian at column " << i << " (eps = " << eps << "): "
                    << h_analytical.norm() << " ("
                    << (h_analytical - h_numerical).norm() << ")" << std::endl;
            }
            if (IsClose(h_numerical, h_analytical, hess_abs_tol, hess_rel_tol)) {
                success = true;
                break;
            }
        }
        Assert(success, error_location, "Hess check failed.");
        if (!success) return false;
    }

    return true;
}

const bool CheckGradientAndHessian(const VectorXr& x0,
    const std::function<const real(const VectorXr&)>& func,
    const std::function<const VectorXr(const VectorXr&)>& func_grad,
    const std::function<const SparseMatrixXr(const VectorXr&)>& func_hess,
    const std::function<const bool(const integer)>& skip_dof,
    const Options& opt) {
    // TODO.
    const std::string error_location = "basic::CheckGradientAndHessian";
    Assert(false, error_location, "Unimplemented.");

    return false;
}

}
