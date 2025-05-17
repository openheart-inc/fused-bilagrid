#include "config.h"


__global__ void bilagrid_uniform_sample_backward_kernel_bilagrid(
    const float* __restrict__ bilagrid,  // [N,12,L,H,W]
    const float* __restrict__ rgb,  // [N,m,h,w,3]
    const float* __restrict__ v_output,  // [N,m,h,w,3]
    float* __restrict__ v_bilagrid,  // [N,12,L,H,W]
    int N, int L, int H, int W,
    int m, int h, int w,
    int num_div
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * 12 * L * H * W * num_div;
    if (idx >= total) return;

    // decode indices
    int div_i = idx % num_div; idx /= num_div;
    int xi = idx % W; idx /= W;
    int yi = idx % H; idx /= H;
    int zi = idx % L; idx /= L;
    int ci = idx % 12; idx /= 12;
    int ni = idx;

    // Channel‐split: si in {0..3}, di in {0..2}
    int si = ci % 4;  // which rgb coeff
    int di = ci / 4;  // which output gradient

    float accum = 0.0f;

    // Loop bounds
    float sw = float(w-1)/float(W-1);
    int wi0 = max((int)ceil((xi-1)*sw), 0);
    int wi1 = min((int)floor((xi+1)*sw), w-1) + 1;
    float sh = float(h-1)/float(H-1);
    int hi0 = max((int)ceil((yi-1)*sh), 0);
    int hi1 = min((int)floor((yi+1)*sh), h-1) + 1;
    int num_px = (hi1-hi0) * (wi1-wi0);
    int pi0 = (num_px * div_i) / num_div;
    int pi1 = (num_px * (div_i+1)) / num_div;

    // Loop over all samples for this batch
    for (int mi = 0; mi < m; ++mi) {
        for (int pi = pi0; pi < pi1; pi++) {
            int hi = hi0 + pi / (wi1-wi0);
            int wi = wi0 + pi % (wi1-wi0);

            // get color
            int g_off = (((ni*m + mi)*h + hi)*w + wi)*3;
            float sr = rgb[g_off+0];
            float sg = rgb[g_off+1];
            float sb = rgb[g_off+2];

            // Flattened sample‐index for coords/rgb/v_output
            float x = (float)wi / (float)(w-1) * (float)(W-1);
            float y = (float)hi / (float)(h-1) * (float)(H-1);
            float z = (kC2G_r * sr + kC2G_g * sg + kC2G_b * sb) * (L-1);

            // Compute corners
            int x0 = floorf(x), y0 = floorf(y), z0 = floorf(z);
            int x1 = x0 + 1,    y1 = y0 + 1,    z1 = z0 + 1;
            z0 = min(max(z0, 0), L-1); z1 = min(max(z1, 0), L-1);
            // if (z != z0 && z != z1) continue;

            // Fractions
            float fx = x-x0, fy = y-y0, fz = z-z0;
            float f000 = (1-fx)*(1-fy)*(1-fz);
            float f001 =    fx*(1-fy)*(1-fz);
            float f010 = (1-fx)*   fy*(1-fz);
            float f011 =    fx*   fy*(1-fz);
            float f100 = (1-fx)*(1-fy)*   fz;
            float f101 =    fx*(1-fy)*   fz;
            float f110 = (1-fx)*   fy*   fz;
            float f111 =    fx*   fy*   fz;

            // Accumulate gradient
            float accum_weight = (
                di == 0 ? v_output[g_off+0]
                : di == 1 ? v_output[g_off+1]
                : v_output[g_off+2]
            ) * (si == 0 ? sr : si == 1 ? sg : si == 2 ? sb : 1.0f);

            float accum_t = 0.0;
            if (zi == z0) {
                if (xi == x0 && yi == y0) accum_t += f000;
                if (xi == x1 && yi == y0) accum_t += f001;
                if (xi == x0 && yi == y1) accum_t += f010;
                if (xi == x1 && yi == y1) accum_t += f011;
            }
            if (zi == z1) {
                if (xi == x0 && yi == y0) accum_t += f100;
                if (xi == x1 && yi == y0) accum_t += f101;
                if (xi == x0 && yi == y1) accum_t += f110;
                if (xi == x1 && yi == y1) accum_t += f111;
            }
            accum += accum_t * accum_weight;
        }
    }

    // Write result
    if (num_div == 1)
        v_bilagrid[((ni*12 + ci)*L + zi)*H*W + yi*W + xi] = accum;
    else
        atomicAdd(v_bilagrid + ((ni*12 + ci)*L + zi)*H*W + yi*W + xi, accum);
}


__global__ void bilagrid_uniform_sample_backward_kernel_rgb(
    const float* __restrict__ bilagrid,  // [N,12,L,H,W]
    const float* __restrict__ rgb,  // [N,m,h,w,3]
    const float* __restrict__ v_output,  // [N,m,h,w,3]
    float* __restrict__ v_rgb,  // [N,m,h,w,3]
    int N, int L, int H, int W,
    int m, int h, int w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * m * h * w;
    if (idx >= total) return;

    int tmp = idx;
    int wi = tmp % w; tmp /= w;
    int hi = tmp % h; tmp /= h;
    int mi = tmp % m; tmp /= m;
    int ni = tmp;

    // input and output colors
    int g_off = (((ni * m + mi) * h + hi) * w + wi) * 3;
    float sr = rgb[g_off+0];
    float sg = rgb[g_off+1];
    float sb = rgb[g_off+2];
    float dr = v_output[g_off+0];
    float dg = v_output[g_off+1];
    float db = v_output[g_off+2]; 
    float vr = 0.0, vg = 0.0, vb = 0.0;

    // grid coords
    float x = (float)wi / (float)(w-1) * (float)(W-1);
    float y = (float)hi / (float)(h-1) * (float)(H-1);
    float z = (kC2G_r * sr + kC2G_g * sg + kC2G_b * sb) * (L-1);
    int x0 = floorf(x), y0 = floorf(y), z0 = floorf(z);
    int x1 = x0 + 1, y1 = y0 + 1, z1 = z0 + 1;
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

    // accumulate bilagrid gradient over 12 channels
    #pragma unroll
    for (int si = 0; si < 3; si++) {
        #pragma unroll
        for (int di = 0; di < 3; di++) {
            int ci = 4 * di + si;
            float gout = (di==0 ? dr : di==1 ? dg : db);

            int base = ((ni*12 + ci)*L*H*W);
            float val =
                bilagrid[base+(z0*H+y0)*W+x0] * w000 +
                bilagrid[base+(z0*H+y0)*W+x1] * w001 +
                bilagrid[base+(z0*H+y1)*W+x0] * w010 +
                bilagrid[base+(z0*H+y1)*W+x1] * w011 +
                bilagrid[base+(z1*H+y0)*W+x0] * w100 +
                bilagrid[base+(z1*H+y0)*W+x1] * w101 +
                bilagrid[base+(z1*H+y1)*W+x0] * w110 +
                bilagrid[base+(z1*H+y1)*W+x1] * w111;
            (si == 0 ? vr : si == 1 ? vg : vb) += val * gout;
        }
    }

    // spatial derivatives for coords
    float dwdz[8] = {
        -(1-fx)*(1-fy), -fx*(1-fy),
        -(1-fx)*fy,     -fx*fy,
         (1-fx)*(1-fy),  fx*(1-fy),
         (1-fx)*fy,      fx*fy
    };

    // accumulate gradient into coords (chain through bilagrid values and rgb)
    float gx_grad = 0.f, gy_grad = 0.f, gz_grad = 0.f;
    #pragma unroll
    for (int corner = 0; corner < 8; ++corner) {
        int xi = (corner & 1) ? x1 : x0;
        int yi = (corner & 2) ? y1 : y0;
        int zi = (corner & 4) ? z1 : z0;
        float trilerp = 0.f;
        // gather the corresponding bilagrid value for each of the 12 channels
        #pragma unroll
        for (int ci = 0; ci < 12; ++ci) {
            const float* vol = bilagrid + ((ni*12 + ci)*L*H*W);
            float v = vol[(zi*H + yi)*W + xi];
            int si = ci % 4, di = ci / 4;
            float r_coeff = (si==0 ? sr : si==1 ? sg : si==2 ? sb : 1.f);
            float gout = (di==0 ? dr : di==1 ? dg : db);
            trilerp += v * r_coeff * gout;
        }
        gz_grad += dwdz[corner] * (L-1) * trilerp;
    }
    v_rgb[g_off+0] = vr + kC2G_r * gz_grad;
    v_rgb[g_off+1] = vg + kC2G_g * gz_grad;
    v_rgb[g_off+2] = vb + kC2G_b * gz_grad;
}


void bilagrid_uniform_sample_backward(
    const float* bilagrid,
    const float* rgb,
    const float* v_output,
    float* v_bilagrid,
    float* v_rgb,
    int N, int L, int H, int W,
    int m, int h, int w
) {
    // v_bilagrid
    {
        int num_div = 1;
        int total = N * 12 * L * H * W * num_div;
        int threads = 64;
        int blocks = (total + threads - 1) / threads;
        bilagrid_uniform_sample_backward_kernel_bilagrid<<<blocks, threads>>>(
            bilagrid, rgb, v_output, v_bilagrid,
            N, L, H, W, m, h, w, num_div
        );
    }

    // v_coords and v_rgb
    {
        int total = N * m * h * w;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        bilagrid_uniform_sample_backward_kernel_rgb<<<blocks, threads>>>(
            bilagrid, rgb, v_output,
            v_rgb,
            N, L, H, W, m, h, w
        );
    }
}
