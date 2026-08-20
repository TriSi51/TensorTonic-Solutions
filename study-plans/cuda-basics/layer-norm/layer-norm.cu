#include <cuda_runtime.h>
#include <math.h>

__global__ void layer_norm_kernel(const float* input, const float* gamma, const float* beta, float* output, int M, int N, float eps) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row > M) {
        return;
    }

    extern __shared__ float shared[];

    float* ssum = shared;
    float* ssumsq = shared + blockDim.x;
    
    float local_sum = 0.0f;
    float local_sumsq = 0.0f;

    for (int i = tid; i < N; i += blockDim.x) {
        int index = row * N + i;
        local_sum += input[index];
        local_sumsq += input[index] * input[index];
    }
    ssum[tid] = local_sum;
    ssumsq[tid] = local_sumsq;
    __syncthreads();

    for (int stride = blockDim.x /2 ; stride > 0 ; stride >>=1) {
        if (tid < stride) {
            ssum[tid] += ssum[tid+stride];
            ssumsq[tid] += ssumsq[tid+stride];
        }
        __syncthreads();
    }

    float all_sum_mean = ssum[0] / N;
    float var = ssumsq[0] / N - all_sum_mean * all_sum_mean;

    float inverse_std = 1.0f/ sqrtf(var + eps);

    for (int col = tid; col < N; col+= blockDim.x) {
        int index = row *N + col;
        output[index] = (input[index] - all_sum_mean) * inverse_std * gamma[col] +beta[col];
    }
    
    
    
}

extern "C" void solve(const float* input, const float* gamma, const float* beta, float* output, int M, int N, float eps) {
    int threads = 256;
    dim3 blocks(M);
    size_t shared_bytes = 2 * threads * sizeof(float);
    layer_norm_kernel<<<blocks, threads, shared_bytes>>>(input, gamma, beta, output, M, N, eps);
    cudaDeviceSynchronize();
}
