#include <cuda_runtime.h>

__global__ void dot_kernel(const float* A, const float* B, float* result, int N) {
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ float shared[];
    float a_i = (i<N) ? A[i] : 0.0f;
    float b_i = (i<N) ? B[i] : 0.0f;

    float product = a_i * b_i;
    float* total_sum = shared;
    total_sum[tid] = product;
    __syncthreads();

    for (int stride = blockDim.x/2 ; stride > 0; stride >>=1) {
        if (tid < stride) {
            total_sum[tid] += total_sum[tid+stride];
            __syncthreads();

        }
    }
    
    if(tid ==0){
        atomicAdd(result, total_sum[0]);
    }
}

extern "C" void solve(const float* A, const float* B, float* result, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    size_t shared_mem = threads * sizeof(float);
    cudaMemset(result, 0, sizeof(float));
    dot_kernel<<<blocks, threads,shared_mem>>>(A, B, result, N);
    cudaDeviceSynchronize();
}
