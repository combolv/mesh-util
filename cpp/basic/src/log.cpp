#include "basic/include/log.hpp"

namespace backend {

void Assert(const bool condition, const std::string& location, const std::string& message) {
    if (!condition) {
        throw std::runtime_error("[" + location + "]" + message);
    }
}

// Timing.
static std::stack<timeval> t_begins;

void Tic() {
    timeval t_begin;
    gettimeofday(&t_begin, nullptr);
    t_begins.push(t_begin);
}

void Toc(const std::string& location, const std::string& message) {
    timeval t_end;
    gettimeofday(&t_end, nullptr);
    timeval t_begin = t_begins.top();
    const real t_interval = (t_end.tv_sec - t_begin.tv_sec) + (t_end.tv_usec - t_begin.tv_usec) / 1e6;
    std::cout << "[" << location << "]: (" << t_interval << "s) " << message << std::endl;
    t_begins.pop();
}

const real Toc() {
    timeval t_end;
    gettimeofday(&t_end, nullptr);
    timeval t_begin = t_begins.top();
    const real t_interval = (t_end.tv_sec - t_begin.tv_sec) + (t_end.tv_usec - t_begin.tv_usec) / 1e6;
    t_begins.pop();

    return t_interval;
}

template<>
void Save<integer>(std::ofstream& f, const integer& val) {
    const int64_t val_cast = static_cast<int64_t>(val);
    f.write(reinterpret_cast<const char*>(&val_cast), sizeof(int64_t));
}

template<>
void Save<real>(std::ofstream& f, const real& val) {
    const double val_cast = static_cast<double>(val);
    f.write(reinterpret_cast<const char*>(&val_cast), sizeof(double));
}

template<>
const integer Load<integer>(std::ifstream& f) {
    int64_t val = 0;
    f.read(reinterpret_cast<char*>(&val), sizeof(int64_t));
    return static_cast<integer>(val);
}

template<>
const real Load<real>(std::ifstream& f) {
    double val = 0;
    f.read(reinterpret_cast<char*>(&val), sizeof(double));
    return static_cast<real>(val);
}

void SaveVectorXr(const std::string& file_name, const VectorXr& vec) {
    std::ofstream f(file_name);
    const integer num = static_cast<integer>(vec.size());
    Save<integer>(f, num);
    for (integer i = 0; i < num; ++i) {
        Save<real>(f, vec(i));
    }
}

const VectorXr LoadVectorXr(const std::string& file_name) {
    std::ifstream f(file_name);
    const integer num = Load<integer>(f);
    VectorXr vec = VectorXr::Zero(num);
    for (integer i = 0; i < num; ++i) {
        vec(i) = Load<real>(f);
    }
    return vec;
}

}