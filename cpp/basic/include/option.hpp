#ifndef GRADY_BASIC_OPTION
#define GRADY_BASIC_OPTION

#include "basic/include/config.hpp"

namespace backend {

template<typename DataType>
class Option {
public:
    Option() : option_() {}
    ~Option() {}

    const DataType& operator[](const std::string& key) const;
    DataType& operator[](const std::string& key);
    const std::map<std::string, DataType>& operator()() const { return option_; }
    void Clear() { return option_.clear(); }

    const bool HasKey(const std::string& key) const;

private:
    std::map<std::string, DataType> option_;
};

template<typename DataType>
std::ostream& operator<<(std::ostream& out, const Option<DataType>& option);

}

#endif