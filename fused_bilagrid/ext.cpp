#include "sample.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bilagrid_sample_forward", &bilagrid_sample_forward_tensor);
    m.def("bilagrid_sample_backward", &bilagrid_sample_backward_tensor);
}
