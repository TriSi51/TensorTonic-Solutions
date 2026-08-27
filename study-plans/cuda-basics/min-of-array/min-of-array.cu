#include <cuda_runtime.h>
#include <float.h>

__global__ void init_result(float* result) {
    result[0] = FLT_MAX;
}

__device__ float atomicMinFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int;
    int assumed;

    do {
        assumed= old;

        float old_value = __int_as_float(assumed);
        if (old_value <= val) {
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
__global__ void min_kernel(const float* input, float* result, int N) {

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ float shared[];

    float x = (i<N) ? input[i] : FLT_MAX;

    shared[tid] = x;
    __syncthreads();

    for (int stride = blockDim.x/2 ; stride > 0; stride >>=1) {
        if (tid < stride) {
            shared[tid] = fminf(shared[tid], shared[tid+stride]);
        }
        __syncthreads();
    }

    if(tid == 0) {
        atomicMinFloat(result, shared[0]);
    }
}

extern "C" void solve(const float* input, float* result, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    size_t shared_mem = threads* sizeof(float);
    init_result<<<1, 1>>>(result);
    min_kernel<<<blocks, threads, shared_mem>>>(input, result, N);
    cudaDeviceSynchronize();
}
