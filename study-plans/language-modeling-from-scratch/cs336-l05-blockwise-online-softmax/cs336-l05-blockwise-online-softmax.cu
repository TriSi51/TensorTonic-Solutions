#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

__device__ __forceinline__ float load_value(
    const void* ptr,
    long long index,
    int dtype_code
) {
    if (dtype_code == 0) {
        return reinterpret_cast<const float*>(ptr)[index];
    } else if (dtype_code == 1) {
        return __half2float(
            reinterpret_cast<const __half*>(ptr)[index]
        );
    } else {
        return __bfloat162float(
            reinterpret_cast<const __nv_bfloat16*>(ptr)[index]
        );
    }
}

__device__ __forceinline__ void store_value(
    void* ptr,
    long long index,
    float value,
    int dtype_code
) {
    if (dtype_code == 0) {
        reinterpret_cast<float*>(ptr)[index] = value;
    } else if (dtype_code == 1) {
        reinterpret_cast<__half*>(ptr)[index] = __float2half_rn(value);
    } else {
        reinterpret_cast<__nv_bfloat16*>(ptr)[index] = __float2bfloat16(value);
    }
}
__global__ void online_softmax_kernel(
    const void* input,
    void* output,
    int rows,
    int cols,
    long long input_row_stride,
    long long input_col_stride,
    long long output_row_stride,
    long long output_col_stride,
    int dtype_code
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= rows) {
        return;
    }


    /*
        shared memory:

        [ max values        ][ sum values       ]
          blockDim.x floats   blockDim.x floats
    */
    extern __shared__ float shared[];

    float* smax = shared;
    float* ssum = shared + blockDim.x;
    /*
        Summary of everything we have processed so far.

        m = current maximum
        l = sum(exp(x - m))
    */
    float m = -CUDART_INF_F;
    float l = 0.0f;

    // Phase 1: process the row block by block and combine(m,l)
    for (int block_start = 0;
        block_start < cols;
        block_start += blockDim.x) {
            
        int col = block_start + tid;
        float x = -CUDART_INF_F;

        if (col < cols) {
            long long idx = 
                static_cast<long long>(row) * input_row_stride +
                static_cast<long long>(col) * input_col_stride;
            x = load_value(input, idx, dtype_code);
        }

        // step 1: find mb = max(current block)
        smax[tid] = x;
        __syncthreads();

        for (int stride = blockDim.x / 2;
            stride > 0;
            stride >>=1) {
            if (tid < stride) {
                smax[tid] = fmaxf(smax[tid], smax[tid + stride]);
            }

            __syncthreads();
        }

        float mb = smax[0];

        // step 2: find sum exp(x_i - mb) for current block
        float local_sum = 0.0f;
        if (col < cols) {
            local_sum  = expf(x-mb);
        }

        ssum[tid] = local_sum;
        __syncthreads();
        for (int stride = blockDim.x / 2;
            stride > 0;
            stride >>=1) {
                
            if  (tid < stride) {
                ssum[tid] += ssum[tid + stride];
            }
            __syncthreads();
        }

        float lb = ssum[0];

        // step 3: combine old (m,l) with block (mb,lb)

        if (tid ==0) {
            float m_new = fmaxf(m,mb);

            float l_new = expf(m - m_new) * l +
                          expf(mb - m_new) * lb;

            smax[0] = m_new;
            ssum[0] = l_new;
        }

        __syncthreads();

        m = smax[0];
        l = ssum[0];

        __syncthreads();


    }
    // PHase 2:  write the softmax
    for (int col = tid;
        col < cols;
        col += blockDim.x) {

        long long input_idx = static_cast<long long>(row) * input_row_stride +
                              static_cast<long long>(col) * input_col_stride;

        long long output_idx = static_cast<long long>(row) * output_row_stride +
                               static_cast<long long>(col) * output_col_stride;

        float x = load_value(
            input,
            input_idx,
            dtype_code
        );


        float y = expf(x-m) / l;

        store_value(
            output,
            output_idx,
            y,
            dtype_code
        );
    }
}

extern "C" void solve(
    const void* input,
    void* output,
    int rows,
    int cols,
    long long input_row_stride,
    long long input_col_stride,
    long long output_row_stride,
    long long output_col_stride,
    int block_size,
    int dtype_code
) {
    if (
        rows < 0 ||
        cols < 0 ||
        block_size < 32 ||
        block_size > 1024 ||
        (block_size & (block_size - 1)) != 0 ||
        dtype_code < 0 ||
        dtype_code > 2
    ) {
        cudaDeviceSynchronize();
        return;
    }
    if (rows == 0 || cols == 0) {
        cudaDeviceSynchronize();
        return;
    }
    size_t shared_bytes = 2 * block_size * sizeof(float);
    online_softmax_kernel<<<rows, block_size, shared_bytes>>>(
        input,
        output,
        rows,
        cols,
        input_row_stride,
        input_col_stride,
        output_row_stride,
        output_col_stride,
        dtype_code
    );
    cudaDeviceSynchronize();
}
