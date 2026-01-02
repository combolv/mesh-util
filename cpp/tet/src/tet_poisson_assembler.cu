#include "tet/include/tet_poisson_assembler.cuh"
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/unique.h>

namespace backend {
namespace tet {

struct COOEntry {
    integer row;
    integer col;
    real    val;
};

struct TetPoissonAssembleFunctor {
    const Vector3r* vertices;
    const Vector4i* tets;
    COOEntry*       entries;

    __device__ inline void compute_gradients(
        const Vector3r& x0,
        const Vector3r& x1,
        const Vector3r& x2,
        const Vector3r& x3,
        Vector3r grad[4],
        real& volume) const
    {
        Vector3r e1 = x1 - x0;
        Vector3r e2 = x2 - x0;
        Vector3r e3 = x3 - x0;

        Vector3r n1 = e2.cross(e3);
        Vector3r n2 = e3.cross(e1);
        Vector3r n3 = e1.cross(e2);

        volume = (e1.dot(n1) / real(6));

        grad[1] = n1 / (real(6) * volume);
        grad[2] = n2 / (real(6) * volume);
        grad[3] = n3 / (real(6) * volume);
        grad[0] = -(grad[1] + grad[2] + grad[3]);
    }

    __device__ void operator()(integer tid) const {
        const Vector4i tet = tets[tid];

        Vector3r x[4];
        for (int i = 0; i < 4; ++i)
            x[i] = vertices[tet(i)];

        Vector3r grad[4];
        real volume;
        compute_gradients(x[0], x[1], x[2], x[3], grad, volume);

        const integer base = tid * 16;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                entries[base + i * 4 + j] = {
                    tet(i),
                    tet(j),
                    volume * grad[i].dot(grad[j])
                };
    }
};

struct ApplyDirichletFlags {
    const uint8_t* is_boundary;
    uint8_t*       is_dirichlet;
    integer        free_vertex;

    __device__ void operator()(integer vid) {
        if (vid == free_vertex)
            is_dirichlet[vid] = 0;
        else
            is_dirichlet[vid] = is_boundary[vid];
    }
};

struct ApplyDirichletToCOO {
    COOEntry* entries;
    const uint8_t* is_dirichlet;

    __device__ void operator()(integer eid) {
        COOEntry& e = entries[eid];

        if (is_dirichlet[e.row]) {
            e.val = (e.row == e.col) ? real(1) : real(0);
        }
        else if (is_dirichlet[e.col]) {
            e.val = real(0);
        }
    }
};

// TODO: pipeline fixes.

    
}
}