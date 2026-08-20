#include <cuda_runtime.h>
#include <math.h>

__global__ void rms_norm_kernel(const float* input, const float* gamma, float* output, int M, int N, float eps) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ float shared[];
    float* ssumsq = shared;

    float local_sumsq = 0.0f;

    for (int j = tid; j < N ; j += blockDim.x) {
        local_sumsq += input[row * N + j] * input[row * N + j];
    }
    ssumsq[tid] = local_sumsq;
    __syncthreads();

    for (int stride = blockDim.x/2 ; stride > 0 ; stride >>=1) {
        if (tid < stride) {
            ssumsq[tid] += ssumsq[tid+stride];
            
        }
        __syncthreads();
        
    }
    float total_sum = ssumsq[0];
    float rms = sqrtf(total_sum / N + eps);

    for (int j = tid; j < N; j += blockDim.x) {
        int index = row * N + j;
        output[index] = input[index] / rms * gamma[j];
    }
    
}

extern "C" void solve(const float* input, const float* gamma, float* output, int M, int N, float eps) {
    int threads = 256;
    dim3 blocks(M);
    size_t shared_bytes =threads * sizeof(float);
    rms_norm_kernel<<<blocks, threads,shared_bytes>>>(input, gamma, output, M, N, eps);
    cudaDeviceSynchronize();
}