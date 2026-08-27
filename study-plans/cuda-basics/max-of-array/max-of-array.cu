#include <cuda_runtime.h>
#include <float.h>

__device__ float atomicMaxFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int;
    int assumed;
    do {
        assumed=  old;
        float old_value = __int_as_float(assumed);
        if ( old_value > val ) {
            break;
        }

        old = atomicCAS(
            address_as_int,
            assumed,
            __float_as_int(val)
        );
        
    } while(old != assumed);
    return __int_as_float(old);
}

__global__ void max_kernel(const float* input, float* result, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    extern __shared__ float shared[];
    float x = (i< N) ? input[i] : -FLT_MAX;

    shared[tid] = x;
    __syncthreads();

    for (int stride = blockDim.x /2; stride > 0; stride >>=1) {
        if (tid < stride) {
            shared[tid] = fmaxf(shared[tid], shared[tid+stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMaxFloat(result, shared[0]);
    }
}

extern "C" void solve(const float* input, float* result, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    float neg_inf = -FLT_MAX;
    size_t shared_mem = threads * sizeof(float);
    cudaMemcpy(result, &neg_inf, sizeof(float), cudaMemcpyHostToDevice);
    max_kernel<<<blocks, threads,shared_mem>>>(input, result, N);
    cudaDeviceSynchronize();
}
