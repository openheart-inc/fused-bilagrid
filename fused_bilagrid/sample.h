#include <torch/extension.h>


void bilagrid_sample_forward(
    const float* bilagrid,
    const float* coords,
    const float* rgb,
    float* output,
    int N, int L, int H, int W,
    int m, int h, int w
);


void bilagrid_sample_backward(
    const float* bilagrid,
    const float* coords,
    const float* rgb,
    const float* v_output,
    float* v_bilagrid,
    float* v_coords,
    float* v_rgb,
    int N, int L, int H, int W,
    int m, int h, int w
);


torch::Tensor bilagrid_sample_forward_tensor(
    torch::Tensor bilagrid, // [N,12,L,H,W]
    torch::Tensor coords,  // [N,m,h,w,3]
    torch::Tensor rgb  // [N,m,h,w,3]
) {
    int N = bilagrid.size(0), L = bilagrid.size(2),
        H = bilagrid.size(3), W = bilagrid.size(4);
    int m = coords.size(1), h = coords.size(2), w = coords.size(3);

    auto output = torch::empty({N, m, h, w, 3}, rgb.options());

    bilagrid_sample_forward(
        bilagrid.data_ptr<float>(),
        coords.data_ptr<float>(),
        rgb.data_ptr<float>(),
        output.data_ptr<float>(),
        N, L, H, W, m, h, w
    );
    
    return output;
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
bilagrid_sample_backward_tensor(
    torch::Tensor bilagrid,  // [N,12,L,H,W]
    torch::Tensor coords,  // [N,m,h,w,3]
    torch::Tensor rgb,  // [N,m,h,w,3]
    torch::Tensor v_output  // [N,m,h,w,3]
) {
    int N = bilagrid.size(0), L = bilagrid.size(2),
        H = bilagrid.size(3), W = bilagrid.size(4);
    int m = coords.size(1), h = coords.size(2), w = coords.size(3);

    auto opts = rgb.options();
    auto v_bilagrid  = torch::zeros({N,12,L,H,W}, opts);
    auto v_coords   = torch::zeros({N,m,h,w,3}, opts);
    auto v_rgb    = torch::zeros({N,m,h,w,3}, opts);

    bilagrid_sample_backward(
        bilagrid.data_ptr<float>(),
        coords.data_ptr<float>(),
        rgb.data_ptr<float>(),
        v_output.data_ptr<float>(),
        v_bilagrid.data_ptr<float>(),
        v_coords.data_ptr<float>(),
        v_rgb.data_ptr<float>(),
        N, L, H, W, m, h, w
    );

    return std::make_tuple(v_bilagrid, v_coords, v_rgb);
}
