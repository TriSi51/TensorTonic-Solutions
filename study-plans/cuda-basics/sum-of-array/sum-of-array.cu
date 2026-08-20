#include <cuda_runtime.h>
#include <stdio.h>

__global__ void sum_kernel(const float* input, float* result, int N) {
    int tid = threadIdx.x;

    extern __shared__ float shared[];

    float* total_sum = shared;
    float sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        sum += input[i];
        
    }
    total_sum[tid]= sum;
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
    sum_kernel<<<1, threads,shared_mems>>>(input, result, N);
    cudaDeviceSynchronize();
}
