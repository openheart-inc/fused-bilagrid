#include <torch/extension.h>


void grid_sample_forward(
    const float* input,
    const float* grid,
    const float* rgb,
    float* output,
    int N, int L, int H, int W,
    int m, int h, int w
);


void grid_sample_backward(
    const float* input,
    const float* grid,
    const float* rgb,
    const float* v_output,
    float* v_input,
    float* v_grid,
    float* v_rgb,
    int N, int L, int H, int W,
    int m, int h, int w
);


torch::Tensor grid_sample_forward_tensor(
    torch::Tensor input, // [N,12,L,H,W]
    torch::Tensor grid,  // [N,m,h,w,3]
    torch::Tensor rgb  // [N,m,h,w,3]
) {
    int N = input.size(0), L = input.size(2),
        H = input.size(3), W = input.size(4);
    int m = grid.size(1), h = grid.size(2), w = grid.size(3);

    auto output = torch::empty({N, m, h, w, 3}, grid.options());

    grid_sample_forward(
        input.data_ptr<float>(),
        grid.data_ptr<float>(),
        rgb.data_ptr<float>(),
        output.data_ptr<float>(),
        N, L, H, W, m, h, w
    );
    
    return output;
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
grid_sample_backward_tensor(
    torch::Tensor input,    // [N,12,L,H,W]
    torch::Tensor grid,     // [N,m,h,w,3]
    torch::Tensor rgb,      // [N,m,h,w,3]
    torch::Tensor v_output  // [N,m,h,w,3]
) {
    int N = input.size(0), L = input.size(2),
        H = input.size(3), W = input.size(4);
    int m = grid.size(1), h = grid.size(2), w = grid.size(3);

    auto opts = input.options();
    auto v_input  = torch::zeros({N,12,L,H,W}, opts);
    auto v_grid   = torch::zeros({N,m,h,w,3}, opts);
    auto v_rgb    = torch::zeros({N,m,h,w,3}, opts);

    grid_sample_backward(
        input.data_ptr<float>(),
        grid.data_ptr<float>(),
        rgb.data_ptr<float>(),
        v_output.data_ptr<float>(),
        v_input.data_ptr<float>(),
        v_grid.data_ptr<float>(),
        v_rgb.data_ptr<float>(),
        N, L, H, W, m, h, w
    );

    return std::make_tuple(v_input, v_grid, v_rgb);
}
