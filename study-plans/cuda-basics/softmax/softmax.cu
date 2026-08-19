#include <cuda_runtime.h>
#include <math_constants.h>

__global__ void softmax_kernel(const float* input, float* output, int N) {
    extern __shared__ float shared[];
    int tid = threadIdx.x;
    
    float smax = -CUDART_INF_F;

    for (int i = tid; i < N; i += blockDim.x) {
        smax = fmaxf(smax, input[i]);
        
    }
    shared[tid] = smax;
    __syncthreads();

    // calculate max val
    for (int i = blockDim.x /2 ; i >0; i>>=1) {
        if (tid < i) {
            shared[tid] = fmaxf(shared[tid], shared[tid+i]);
        }

        __syncthreads();
    }

    smax = shared[0];

    // cal sum
    float local_sum = 0.0f;

    for (int i = tid; i < N; i += blockDim.x) {
        local_sum += expf(input[i]- smax);
    }

    shared[tid] = local_sum;
    __syncthreads();

    for (int i = blockDim.x/2; i >0 ; i >>=1) {
        if (tid < i){
            shared[tid] += shared[tid + i];
        }
        __syncthreads();
    }
    local_sum = shared[0];
    // cal softmax
    for (int i = tid; i < N; i += blockDim.x) {
        output[i] = expf(input[i] - smax) / local_sum;
    }
    
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    softmax_kernel<<<1, threads, threads* sizeof(float)>>>(input, output, N);
    cudaDeviceSynchronize();
}