#include <cuda_runtime.h>
#include <float.h>

__global__ void argmax_kernel(const float* input, float* block_vals, int* block_idxs, int N) {
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float s_vals[256];
    __shared__ float s_idxs[256];

    if (i<N ) {
        s_vals[tid] = input[i];
        s_idxs[tid] = i;
    } else {
        s_vals[tid] = -FLT_MAX;
        s_idxs[tid] = -1;
    }
    __syncthreads();
    for (int stride = blockDim.x/2; stride > 0; stride >>=1) {
        if (tid < stride) {
            float left_val = s_vals[tid];
            float right_val = s_vals[tid+stride];

            float left_idx = s_idxs[tid];
            float right_idx = s_idxs[tid+stride];

            if (right_val > left_val || (left_val == right_val && right_idx < left_idx)) {
                s_vals[tid] = right_val;
                s_idxs[tid] = right_idx;
            }

        }
        __syncthreads();
    }

    if (tid == 0) {
        block_vals[blockIdx.x] = s_vals[0];
        block_idxs[blockIdx.x] = s_idxs[0];
    }
}

__global__ void argmax_finalize_kernel(const float* block_vals, const int* block_idxs, int* result, int num_blocks) {
    int tid = threadIdx.x;

    __shared__ float s_vals[256];
    __shared__ int s_idxs[256];

    float local_max = -FLT_MAX;
    int local_idx = -1;

    for (int i = tid; i <num_blocks; i += blockDim.x) {
        float val = block_vals[i];
        int idx = block_idxs[i];

        if (val > local_max || (val == local_max && idx < local_idx)){
            local_max = val;
            local_idx = idx;
        }

    }
    s_vals[tid] = local_max;
    s_idxs[tid] = local_idx;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0 ; stride >>=1) {
        if (tid< stride) {
            float left_val = s_vals[tid];
            float right_val = s_vals[tid+stride];

            float left_idx = s_idxs[tid];
            float right_idx = s_idxs[tid+stride];

            if (right_val > left_val || (left_val == right_val && right_idx < left_idx)) {
                s_vals[tid] = right_val;
                s_idxs[tid] = right_idx;
            }
        }
        __syncthreads();
    }
    if (tid == 0)  {
        result[0] = s_idxs[0];
    }
    
}

extern "C" void solve(const float* input, int* result, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    float* block_vals = nullptr;
    int* block_idxs = nullptr;
    cudaMalloc(&block_vals, blocks * sizeof(float));
    cudaMalloc(&block_idxs, blocks * sizeof(int));

    argmax_kernel<<<blocks, threads>>>(input, block_vals, block_idxs, N);
    argmax_finalize_kernel<<<1, threads>>>(block_vals, block_idxs, result, blocks);
    cudaDeviceSynchronize();

    cudaFree(block_vals);
    cudaFree(block_idxs);
}
