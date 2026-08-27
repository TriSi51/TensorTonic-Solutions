#include <cuda_runtime.h>
#include <float.h>

__global__ void argmin_kernel(const float* input, float* block_vals, int* block_idxs, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    __shared__ float vals[256];
    __shared__ int idxs[256];

    if (i < N ) {
        vals[tid] = input[i];
        idxs[tid] = i;
    } else{
        vals[tid]= FLT_MAX;
        idxs[tid] = -1;
    }

    __syncthreads();

    for (int stride = blockDim.x/2; stride > 0; stride >>=1) {
        if (tid < stride) {
            float left_val = vals[tid];
            float right_val = vals[tid+stride];

            float left_idx = idxs[tid];
            float right_idx = idxs[tid+stride];

            if (right_val < left_val || (right_val == left_val && right_idx < left_idx)) {
                vals[tid] = right_val;
                idxs[tid] = right_idx;
            }
        }

        __syncthreads();
    }

    if (tid ==0 ) {
        block_vals[blockIdx.x] = vals[0];
        block_idxs[blockIdx.x] = idxs[0];
    }
}

__global__ void argmin_finalize_kernel(const float* block_vals, const int* block_idxs, int* result, int num_blocks) {
    int tid = threadIdx.x;

    __shared__ float s_vals[256];
    __shared__ float s_idxs[256];


    float local_min = FLT_MAX;
    int local_idx = -1;
    for (int i = tid; i < num_blocks; i += blockDim.x) {
        float val = block_vals[i];
        float idx = block_idxs[i];

        if (val < local_min || (val == local_min && idx < local_idx)) {
            local_min = val;
            local_idx = idx;
        }
    }
    s_vals[tid]= local_min;
    s_idxs[tid] = local_idx;
    __syncthreads();

    for (int stride = blockDim.x /2 ; stride > 0; stride >>=1) {
        if (tid < stride) {
            float left_val = s_vals[tid];
            float right_val = s_vals[tid+stride];

            int left_idx = s_idxs[tid];
            int right_idx = s_idxs[tid+ stride];

            if (right_val < left_val || (right_val == left_val && right_idx < left_idx)) {
                s_vals[tid]= right_val;
                s_idxs[tid]= right_idx;
            }
            
        }

        __syncthreads();
    }
    if (tid == 0) {
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

    argmin_kernel<<<blocks, threads>>>(input, block_vals, block_idxs, N);
    argmin_finalize_kernel<<<1, threads>>>(block_vals, block_idxs, result, blocks);
    cudaDeviceSynchronize();

    cudaFree(block_vals);
    cudaFree(block_idxs);
}
