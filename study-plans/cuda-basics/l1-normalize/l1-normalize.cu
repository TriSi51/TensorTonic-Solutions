#include <cuda_runtime.h>
#include <math.h>

__global__ void l1_sum_kernel(
    const float* input,
    float* sum,
    int N
) {
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ float shared[];

    float x = (i<N) ? fabsf(input[i]) : 0.0f;

    shared[tid] = x;
    __syncthreads();

    for (int stride = blockDim.x /2; stride > 0; stride >>=1) {
        if (tid < stride) {
            shared[tid] += shared[tid+stride];
        }

        __syncthreads();
    }

    if (tid ==0) {
        atomicAdd(sum, shared[0]);
    }
    
}

__global__ void l1_normalize_kernel(const float* input, float* output, const float* sum, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i < N) {
        float denominator = *sum;

        if (denominator != 0.0f) {
            output[i] = input[i] / denominator;
        }
        else {
            output[i] = 0.0f;
        }
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    float* d_sum;
    cudaMalloc(&d_sum, sizeof(float));

    cudaMemset(d_sum, 0, sizeof(float));
    size_t shared_mem = threads * sizeof(float);
    
    l1_sum_kernel<<<blocks, threads, shared_mem>>>(
        input,
        d_sum,
        N
    );
    l1_normalize_kernel<<<blocks, threads>>>(input, output, d_sum, N);
    cudaDeviceSynchronize();
    cudaFree(d_sum);
}
