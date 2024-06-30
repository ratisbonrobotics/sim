#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define BLOCK_SIZE 256
#define GRID_SIZE 256

__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) 
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__device__ __forceinline__ float blockReduceSum(float val) {
    static __shared__ float shared[32]; // Shared mem for 32 partial sums
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warpReduceSum(val);     // Each warp performs partial reduction

    if (lane == 0) shared[wid] = val; // Write reduced value to shared memory

    __syncthreads();              // Wait for all partial reductions

    // Read from shared memory only if that warp existed
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

    if (wid == 0) val = warpReduceSum(val); // Final reduce within first warp

    return val;
}

__global__ void reduceSum(float *g_idata, float *g_odata, unsigned int n) {
    float sum = 0;
    
    // Grid-stride loop, so each thread block handles multiple elements
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        sum += g_idata[i];
    }
    
    sum = blockReduceSum(sum);
    
    if (threadIdx.x == 0) {
        atomicAdd(g_odata, sum);
    }
}

__global__ void finalReduceSum(float *g_idata, float *g_odata, unsigned int n) {
    float sum = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        sum += g_idata[i];
    }
    
    sum = blockReduceSum(sum);
    
    if (threadIdx.x == 0) {
        *g_odata = sum;
    }
}

float sumArray(float* h_idata, int size) {
    float *d_idata, *d_odata;
    cudaMalloc((void**)&d_idata, size * sizeof(float));
    cudaMalloc((void**)&d_odata, sizeof(float));

    cudaMemcpy(d_idata, h_idata, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_odata, 0, sizeof(float));

    int numBlocks = GRID_SIZE;
    int threadsPerBlock = BLOCK_SIZE;

    reduceSum<<<numBlocks, threadsPerBlock>>>(d_idata, d_odata, size);
    finalReduceSum<<<1, threadsPerBlock>>>(d_odata, d_odata, 1);

    float sum;
    cudaMemcpy(&sum, d_odata, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_idata);
    cudaFree(d_odata);

    return sum;
}

int main() {
    const int SIZE = 1 << 24;  // 16M elements
    float* h_idata = (float*)malloc(SIZE * sizeof(float));

    for (int i = 0; i < SIZE; i++) {
        h_idata[i] = 1.0f;
    }

    float sum = sumArray(h_idata, SIZE);

    printf("Sum: %f\n", sum);

    free(h_idata);

    return 0;
}