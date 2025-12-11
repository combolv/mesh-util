#ifndef TET_CONNECTED_COMPONENT_HPP
#define TET_CONNECTED_COMPONENT_HPP
#include "basic/include/log.hpp"

namespace backend {
namespace tet {

struct ConnectedComponentExtractResult {
    bool is_connected;          // true if the whole mesh was one component
    integer num_components;         // number of connected components (by vertex connectivity)
    integer largest_component_root; // representative root index (vertex id) of the chosen component
    real largest_bbox_diag;   // diagonal length of chosen component's AABB
    std::vector<integer> comp_sizes;// sizes (# vertices) of each component (indexed implicitly by order-of-discovery)
};

// Simple Union-Find / Disjoint Set (path compression + union by rank)
struct DSU {
    std::vector<integer> parent;
    std::vector<integer> rank;
    DSU(const integer n) { 
        parent.resize(n);
        rank.assign(n, 0);
        for (integer i = 0; i < n; ++i) parent[i] = i;
    }
    integer find(const integer x) {
        integer p = parent[x];
        if (p != x) parent[x] = find(p);
        return parent[x];
    }
    void unite(const integer a, const integer b) {
        integer ra = find(a), rb = find(b);
        if (ra == rb) return;
        if (rank[ra] < rank[rb]) parent[ra] = rb;
        else if (rank[rb] < rank[ra]) parent[rb] = ra;
        else { parent[rb] = ra; ++rank[ra]; }
    }
};

const ConnectedComponentExtractResult ExtractLargestComponentByAABB(const Matrix3Xr&, const Matrix4Xi&, Matrix3Xr&, Matrix4Xi&);

class TetMeshConnectedComponentExtractor {
public:
    TetMeshConnectedComponentExtractor(const Matrix3Xr& vertices, const Matrix4Xi& elements) :
        vertices_(vertices), elements_(elements) {}
    
    void Compute() { result_ = ExtractLargestComponentByAABB(vertices_, elements_, Vout_, Tout_); computed_ = true; }

    const std::pair<Matrix3Xr, Matrix4Xi> GetExtractedMesh() const {
        Assert(computed_, "tet::TetMeshConnectedComponentExtractor::GetExtractedMesh", "Compute() must be called before GetExtractedMesh().");
        return {Vout_, Tout_};
    }

    const bool IsConnected() const {
        Assert(computed_, "tet::TetMeshConnectedComponentExtractor::IsConnected", "Compute() must be called before IsConnected().");
        return result_.is_connected;
    }

    const std::string GetReportString() const {
        Assert(computed_, "tet::TetMeshConnectedComponentExtractor::GetReportString", "Compute() must be called before GetReportString().");
        std::ostringstream oss;
        oss << "Connected Component Extraction Report:\n";
        oss << "  Is Connected: " << (result_.is_connected ? "Yes" : "No") << "\n";
        oss << "  Number of Components: " << result_.num_components << "\n";
        oss << "  Largest Component Root Vertex ID: " << result_.largest_component_root << "\n";
        oss << "  Largest Component AABB Diagonal Length: " << result_.largest_bbox_diag << "\n";
        oss << "  Component Sizes (in #vertices): ";
        for (size_t i = 0; i < result_.comp_sizes.size(); ++i) {
            oss << result_.comp_sizes[i];
            if (i + 1 < result_.comp_sizes.size()) oss << ", ";
        }
        oss << "\n";
        return oss.str();
    }

private:
    bool computed_ = false;
    const Matrix3Xr vertices_;
    const Matrix4Xi elements_;
    ConnectedComponentExtractResult result_;
    Matrix3Xr Vout_;
    Matrix4Xi Tout_;
};

}
}

#endif