#include <cuda_runtime.h>

__global__ void mean_variance_kernel(const float* input, float* mean_out, float* var_out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    extern __shared__ float shared[];
    float* ssum = shared;
    float* ssumsq = shared + blockDim.x;

    float sum = 0.0f;
    float sumsq= 0.0f;
    float x = (i<N) ? input[i] : 0.0f;
    ssum[tid]= x;
    ssumsq[tid] = x * x;

    __syncthreads();
    for (int stride = blockDim.x /2 ; stride >0; stride >>=1) {
        if (tid <stride) {
            ssum[tid] += ssum[tid+stride];
            ssumsq[tid] += ssumsq[tid+stride];
        }
        __syncthreads();
    
    }

    
    
    
    if (tid ==0){
        atomicAdd(mean_out, ssum[0]);    
        atomicAdd(var_out, ssumsq[0]);     
    }
    
}

__global__ void finalize_kernel(
    float* mean_out,
    float* var_out,
    int N
) {
    mean_out[0] = mean_out[0] / N;
    var_out[0] = var_out[0] /N - mean_out[0] * mean_out[0];
}

extern "C" void solve(const float* input, float* mean_out, float* var_out, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    size_t shared_mem = 2 * threads * sizeof(float);
    cudaMemset(mean_out, 0, sizeof(float));
    cudaMemset(var_out, 0, sizeof(float));
    mean_variance_kernel<<<blocks, threads, shared_mem>>>(input, mean_out, var_out, N);
    finalize_kernel<<<1,1>>>(mean_out, var_out, N);
    cudaDeviceSynchronize();
}
