#ifndef BACKEND_BASIC_LOG
#define BACKEND_BASIC_LOG

#include "basic/include/config.hpp"

namespace backend {

// Timing.
void Tic();
void Toc(const std::string& location, const std::string& message);
const real Toc();

// Load and save sparse matrices.
template<typename DataType>
void Save(std::ofstream& f, const DataType& val);
template<typename DataType>
const DataType Load(std::ifstream& f);

void SaveVectorXr(const std::string& file_name, const VectorXr& vec);
const VectorXr LoadVectorXr(const std::string& file_name);

void Assert(const bool condition, const std::string& location, const std::string& message);

}

#endif