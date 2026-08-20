#include <cuda_runtime.h>
#include <stdio.h>

__global__ void sum_kernel(const float* input, float* result, int N) {
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float x = (i<N) ? input[i] : 0.0f;
    extern __shared__ float shared[];

    float* total_sum = shared;
    
    total_sum[tid]= x;
    __syncthreads();
    for (int stride = blockDim.x/2 ; stride > 0; stride >>=1){
        if (tid < stride) {
            total_sum[tid] += total_sum[tid+stride];
        }
        __syncthreads();
    }
    
    if (tid ==0) {
        atomicAdd(result,total_sum[0]);    
    }
    
}

extern "C" void solve(const float* input, float* result, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    size_t shared_mems = threads * sizeof(float);
    cudaMemset(result, 0, sizeof(float));
    sum_kernel<<<blocks, threads,shared_mems>>>(input, result, N);
    cudaDeviceSynchronize();
}
