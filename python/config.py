# Extract the real and integer types from the C++ code.
from pathlib import Path
import numpy as np
# from grady_root import root

def extract_data_types():
    # config_file = Path(root) / "cpp" / "basic" / "include" / "config.hpp"
    # np_real = None
    # np_integer = None
    # with open(config_file, "r") as f:
    #     lines = f.readlines()
    #     for l in lines:
    #         if "using integer = int32_t;" in l:
    #             np_integer = np.int32
    #         elif "using integer = int64_t;" in l:
    #             np_integer = np.int64
    #         elif "using real = float;" in l:
    #             np_real = np.float32
    #         elif "using real = double;" in l:
    #             np_real = np.float64
    np_integer = np.int32
    np_real = np.float64
    assert np_real is not None and np_integer is not None
    return np_integer, np_real

np_integer, np_real = extract_data_types()

def to_real_array(val):
    return np.array(val, dtype=np_real).copy()

def to_integer_array(val):
    return np.array(val, dtype=np_integer).copy()