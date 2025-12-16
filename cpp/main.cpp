#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/eigen.h"
#include "tet/include/tet_cross_section.cuh"
#include "tet/include/tet_surface_mesh_extractor.cuh"
#include "tet/include/tet_connected_component.hpp"


PYBIND11_MODULE(backend, m) {
    m.def("ComputeTetXYIntersectionGpu", &backend::tet::ComputeTetXYIntersectionGpu, pybind11::arg("tet_vertices"), pybind11::arg("tet_indices"), pybind11::arg("tet_scalar_field"));
    pybind11::class_<backend::tet::TetMeshConnectedComponentExtractor>(m, "TetMeshConnectedComponentExtractor")
        .def(pybind11::init<const backend::Matrix3Xr&, const backend::Matrix4Xi&>(), pybind11::arg("vertices"), pybind11::arg("elements"))
        .def("Compute", &backend::tet::TetMeshConnectedComponentExtractor::Compute)
        .def("GetExtractedMesh", &backend::tet::TetMeshConnectedComponentExtractor::GetExtractedMesh)
        .def("IsConnected", &backend::tet::TetMeshConnectedComponentExtractor::IsConnected)
        .def("GetReportString", &backend::tet::TetMeshConnectedComponentExtractor::GetReportString)
        .def("GetComponentSizes", &backend::tet::TetMeshConnectedComponentExtractor::GetComponentSizes);
    pybind11::class_<backend::tet::SurfaceMeshConnectedComponentExtractor>(m, "SurfaceMeshConnectedComponentExtractor")
        .def(pybind11::init<const backend::Matrix3Xr&, const backend::Matrix3Xi&>(), pybind11::arg("vertices"), pybind11::arg("faces"))
        .def("Compute", &backend::tet::SurfaceMeshConnectedComponentExtractor::Compute)
        .def("GetExtractedMesh", &backend::tet::SurfaceMeshConnectedComponentExtractor::GetExtractedMesh)
        .def("IsConnected", &backend::tet::SurfaceMeshConnectedComponentExtractor::IsConnected)
        .def("GetReportString", &backend::tet::SurfaceMeshConnectedComponentExtractor::GetReportString);
    pybind11::class_<backend::tet::TetSurfaceMeshExtractor>(m, "TetSurfaceMeshExtractor")
        .def(pybind11::init<const backend::Matrix3Xr&, const backend::Matrix4Xi&>(), pybind11::arg("vertices"), pybind11::arg("elements"))
        .def("Compute", &backend::tet::TetSurfaceMeshExtractor::Compute)
        .def("GetSurfaceMesh", &backend::tet::TetSurfaceMeshExtractor::GetSurfaceMesh)
        .def("GetOldFaces", &backend::tet::TetSurfaceMeshExtractor::GetOldFaces)
        .def("GetParentTet", &backend::tet::TetSurfaceMeshExtractor::GetParentTet);
}