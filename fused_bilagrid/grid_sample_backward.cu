__global__ void grid_sample_backward_kernel(
    const float* __restrict__ input,    // [N,12,L,H,W]
    const float* __restrict__ grid,     // [N,m,h,w,3]
    const float* __restrict__ rgb,      // [N,m,h,w,3]
    const float* __restrict__ v_output, // [N,m,h,w,3]
    float* __restrict__ v_input,        // [N,12,L,H,W]
    float* __restrict__ v_grid,         // [N,m,h,w,3]
    float* __restrict__ v_rgb,          // [N,m,h,w,3]
    int N, int L, int H, int W,
    int m, int h, int w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * m * h * w;
    if (idx >= total) return;

    // decode indices
    int tmp = idx;
    int wi = tmp % w; tmp /= w;
    int hi = tmp % h; tmp /= h;
    int mi = tmp % m; tmp /= m;
    int ni = tmp;    

    // grid coords in [-1,1]
    int g_off = (((ni*m + mi)*h + hi)*w + wi)*3;
    float gx = grid[g_off+0], gy = grid[g_off+1], gz = grid[g_off+2];
    // map to volume coords
    float x = ((gx + 1.f)*0.5f)*(W - 1);
    float y = ((gy + 1.f)*0.5f)*(H - 1);
    float z = ((gz + 1.f)*0.5f)*(L - 1);

    // floor + ceil, clamped
    int x0 = floorf(x), y0 = floorf(y), z0 = floorf(z);
    int x1 = x0 + 1, y1 = y0 + 1, z1 = z0 + 1;
    x0 = min(max(x0,0), W-1); x1 = min(max(x1,0), W-1);
    y0 = min(max(y0,0), H-1); y1 = min(max(y1,0), H-1);
    z0 = min(max(z0,0), L-1); z1 = min(max(z1,0), L-1);

    // fractional parts
    float fx = x - x0, fy = y - y0, fz = z - z0;
    float w000 = (1-fx)*(1-fy)*(1-fz);
    float w001 = fx*(1-fy)*(1-fz);
    float w010 = (1-fx)*fy*(1-fz);
    float w011 = fx*fy*(1-fz);
    float w100 = (1-fx)*(1-fy)*fz;
    float w101 = fx*(1-fy)*fz;
    float w110 = (1-fx)*fy*fz;
    float w111 = fx*fy*fz;

    // read rgb coeffs and upstream gradient
    float sr = rgb[g_off+0], sg = rgb[g_off+1], sb = rgb[g_off+2];
    float dr = v_output[g_off+0];
    float dg = v_output[g_off+1];
    float db = v_output[g_off+2];

    // accumulate input gradient over 12 channels
    #pragma unroll
    for (int ci = 0; ci < 12; ++ci) {
        // weight from rgb channel
        int si = ci % 4, di = ci / 4;
        float r_coeff = (si==0 ? sr : si==1 ? sg : si==2 ? sb : 1.f);
        float gout = (di==0 ? dr : di==1 ? dg : db);

        // scatter back into the eight corners
        int base = ((ni*12 + ci)*L*H*W);
        atomicAdd(v_input + base + (z0*H + y0)*W + x0, w000 * r_coeff * gout);
        atomicAdd(v_input + base + (z0*H + y0)*W + x1, w001 * r_coeff * gout);
        atomicAdd(v_input + base + (z0*H + y1)*W + x0, w010 * r_coeff * gout);
        atomicAdd(v_input + base + (z0*H + y1)*W + x1, w011 * r_coeff * gout);
        atomicAdd(v_input + base + (z1*H + y0)*W + x0, w100 * r_coeff * gout);
        atomicAdd(v_input + base + (z1*H + y0)*W + x1, w101 * r_coeff * gout);
        atomicAdd(v_input + base + (z1*H + y1)*W + x0, w110 * r_coeff * gout);
        atomicAdd(v_input + base + (z1*H + y1)*W + x1, w111 * r_coeff * gout);

        // gradient w.r.t. rgb coefficients
        float val = 
            ( ( (input[base + (z0*H + y0)*W + x0]*(1-fx) + input[base + (z0*H + y0)*W + x1]*fx)*(1-fy)
              + (input[base + (z0*H + y1)*W + x0]*(1-fx) + input[base + (z0*H + y1)*W + x1]*fx)*fy )*(1-fz)
            + ( (input[base + (z1*H + y0)*W + x0]*(1-fx) + input[base + (z1*H + y0)*W + x1]*fx)*(1-fy)
              + (input[base + (z1*H + y1)*W + x0]*(1-fx) + input[base + (z1*H + y1)*W + x1]*fx)*fy )*fz
            );
        atomicAdd(v_rgb + g_off + si, val * gout);
    }

    // spatial derivatives for grid
    // ∂w000/∂x = -(1-fy)*(1-fz), ∂w001/∂x = +(1-fy)*(1-fz), etc...
    float dwdx[8] = {
        -(1-fy)*(1-fz),  (1-fy)*(1-fz),
        -fy*(1-fz),      fy*(1-fz),
        -(1-fy)*fz,      (1-fy)*fz,
        -fy*fz,          fy*fz
    };
    float dwdy[8] = {
        -(1-fx)*(1-fz), -fx*(1-fz),
         (1-fx)*(1-fz),  fx*(1-fz),
        -(1-fx)*fz,     -fx*fz,
         (1-fx)*fz,      fx*fz
    };
    float dwdz[8] = {
        -(1-fx)*(1-fy), -fx*(1-fy),
        -(1-fx)*fy,     -fx*fy,
         (1-fx)*(1-fy),  fx*(1-fy),
         (1-fx)*fy,      fx*fy
    };

    // accumulate gradient into grid (chain through input values and rgb)
    float gx_grad = 0.f, gy_grad = 0.f, gz_grad = 0.f;
    #pragma unroll
    for (int corner = 0; corner < 8; ++corner) {
        int xi = (corner & 1) ? x1 : x0;
        int yi = (corner & 2) ? y1 : y0;
        int zi = (corner & 4) ? z1 : z0;
        float trilerp = 0.f;
        // gather the corresponding input value for each of the 12 channels
        #pragma unroll
        for (int ci = 0; ci < 12; ++ci) {
            const float* vol = input + ((ni*12 + ci)*L*H*W);
            float v = vol[(zi*H + yi)*W + xi];
            int si = ci % 4, di = ci / 4;
            float r_coeff = (si==0 ? sr : si==1 ? sg : si==2 ? sb : 1.f);
            float gout = (di==0 ? dr : di==1 ? dg : db);
            trilerp += v * r_coeff * gout;
        }
        gx_grad += dwdx[corner] * (W-1)*0.5f * trilerp;
        gy_grad += dwdy[corner] * (H-1)*0.5f * trilerp;
        gz_grad += dwdz[corner] * (L-1)*0.5f * trilerp;
    }
    atomicAdd(v_grid + g_off + 0, gx_grad);
    atomicAdd(v_grid + g_off + 1, gy_grad);
    atomicAdd(v_grid + g_off + 2, gz_grad);
}


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
) {
    int total = N * m * h * w;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    grid_sample_backward_kernel<<<blocks, threads>>>(
        input, grid, rgb, v_output,
        v_input, v_grid, v_rgb,
        N, L, H, W, m, h, w
    );
}
