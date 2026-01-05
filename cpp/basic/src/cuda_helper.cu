#include "basic/include/cuda_helper.hpp"
#include "basic/include/log.hpp"
#include "basic/include/math.hpp"

// For now, the PrintInfo functions are disabled by the following micros.
#define PrintInfo(...) ;

namespace backend {

void PrintCudaVersionInfo() {
    const std::string location = "basic::cuda_helper::PrintCudaVersionInfo";

    // We should not use integer here because cudaRuntimeGetVersion explicitly requires int as input.
    int runtime_version;
    cudaRuntimeGetVersion(&runtime_version);
    PrintInfo(location, "CUDA runtime version: " + std::to_string(runtime_version) + ".");

    int driver_version;
    cudaDriverGetVersion(&driver_version);
    PrintInfo(location, "CUDA driver version: " + std::to_string(driver_version) + ".");
}

// https://github.com/stephen-sorley/cuda-cmake.
__global__ void GpuAddKernel(const integer num, real* x, real* y) {
    const integer thread_grid_idx = static_cast<integer>(blockIdx.x * blockDim.x + threadIdx.x);
    const integer num_threads_in_grid = static_cast<integer>(blockDim.x * gridDim.x);
    for (integer i = thread_grid_idx; i < num; i += num_threads_in_grid) y[i] += x[i];
}

void CpuAdd(const integer num) {
    std::vector<real> x(num, 1);
    std::vector<real> y(num, 2);

    Tic();
    for (integer i = 0; i < num; ++i) y[i] += x[i];
    Toc("basic::cuda_helper::CpuAdd", "Summing " + std::to_string(num) + " real numbers on a CPU");
}

void GpuAdd(const integer num) {
    std::vector<real> x(num, 1);
    std::vector<real> y(num, 2);

    Tic();
    real* d_x = nullptr;
    real* d_y = nullptr;

    cudaMalloc(&d_x, num * sizeof(real));
    cudaMalloc(&d_y, num * sizeof(real));


    // Copy vectors to device.
    cudaMemcpy(d_x, x.data(), num * sizeof(real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y.data(), num * sizeof(real), cudaMemcpyHostToDevice);

    // Perform the addition.
    // GTX 1080.
    const integer num_sm = 20;
    const integer blocks_per_sm = 16;
    const integer threads_per_block = 1024;
    GpuAddKernel<<<blocks_per_sm * num_sm, threads_per_block>>>(num, d_x, d_y);

    // Copy results back from device.
    cudaMemcpy(y.data(), d_y, num * sizeof(real), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_y);

    Toc("basic::cuda_helper::GpuAdd", "Summing " + std::to_string(num) + " real numbers on a GPU");

    // Verify that returned results are OK.
    real max_err = 0;
    for (integer i = 0; i < num; ++i) max_err = std::max(max_err, std::abs(y[i] - 3));
    Assert(IsClose(max_err, 0, ToReal(1e-8), ToReal(1e-8)), "basic::cuda_helper::GpuAdd", "Inaccurate results from the GPU.");
}

void CheckCudaStatus(const std::string& error_location) {
    Assert(cudaGetLastError() == cudaError::cudaSuccess, error_location, cudaGetErrorString(cudaGetLastError()));
}

}