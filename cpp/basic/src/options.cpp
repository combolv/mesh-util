#include "basic/include/options.hpp"

namespace backend {

void Options::Clear() {
    integer_option_.Clear();
    real_option_.Clear();
    string_option_.Clear();
}

}