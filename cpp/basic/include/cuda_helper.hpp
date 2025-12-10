#ifndef GRADY_BASIC_CUDA_HELPER
#define GRADY_BASIC_CUDA_HELPER

#include "basic/include/log.hpp"

namespace backend {

void PrintCudaVersionInfo();

void CpuAdd(const integer num);
void GpuAdd(const integer num);
void CheckCudaStatus(const std::string& error_location);


}

#endif