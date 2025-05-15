#include "grid_sample.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("grid_sample_forward", &grid_sample_forward_tensor);
    m.def("grid_sample_backward", &grid_sample_backward_tensor);
}
