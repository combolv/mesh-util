#ifndef GRADY_BASIC_OPTIONS
#define GRADY_BASIC_OPTIONS

#include "basic/include/option.hpp"

namespace backend {

class Options {
public:
    Options() {}
    ~Options() {}

    const Option<integer>& integer_option() const { return integer_option_; }
    Option<integer>& integer_option() { return integer_option_; }
    const Option<real>& real_option() const { return real_option_; }
    Option<real>& real_option() { return real_option_; }
    const Option<std::string>& string_option() const { return string_option_; }
    Option<std::string>& string_option() { return string_option_; }

    void Clear();

private:
    Option<integer> integer_option_;
    Option<real> real_option_;
    Option<std::string> string_option_;
};

}

#endif