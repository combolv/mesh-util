#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/eigen.h"
#include "tet/include/tet_visualizer.cuh"

PYBIND11_MODULE(backend, m) {
    m.def("ComputeTetXYIntersectionGpu", &backend::tet::ComputeTetXYIntersectionGpu, pybind11::arg("tet_vertices"), pybind11::arg("tet_indices"), pybind11::arg("tet_scalar_field"));
}