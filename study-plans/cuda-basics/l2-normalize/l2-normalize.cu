#include <cuda_runtime.h>
#include <math.h>

__global__ void reduce_sq_sum(const float* input, float* sumv, int N) {

    int tid = threadIdx.x;

    extern __shared__ float shared[];
    float sumsq = 0.0f;
    for (int i = tid; i < N; i+= blockDim.x) {
        sumsq += input[i] * input[i]; 
    }

    shared[tid] = sumsq;
    __syncthreads();
    for (int stride = blockDim.x/2 ; stride > 0; stride >>=1) {
        if (tid < stride) {
            shared[tid] += shared[tid+ stride];
        }
        __syncthreads();
    }
    if (tid ==0) {
        atomicAdd(sumv, shared[0]);
    }
    
}

__global__ void divide_by_sqrt(const float* input, float* output, const float* sumv, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i < N) {
        float denominator = *sumv;
        if (denominator != 0.0f) {
            output[i] = input[i] * rsqrtf(denominator);
        }
        else {
            output[i] = 0.0f;
        }
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    float* d_sum;
    cudaMalloc(&d_sum, sizeof(float));
    cudaMemset(d_sum, 0, sizeof(float));
    int threads = 256;
    size_t shared_mem = threads * sizeof(float);
    reduce_sq_sum<<<1, 256,shared_mem>>>(input, d_sum, N);
    
    
    int blocks = (N + threads - 1) / threads;
    divide_by_sqrt<<<blocks, threads>>>(input, output, d_sum, N);

    cudaDeviceSynchronize();
    cudaFree(d_sum);
}
