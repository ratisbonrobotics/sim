#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define WIDTH 8192
#define HEIGHT 8192
#define MAX_ITERATIONS 1000
#define BLOCK_SIZE 256

__device__ __forceinline__ int mandelbrot(float c_re, float c_im) {
    float z_re = c_re, z_im = c_im;
    float z_re2 = z_re * z_re, z_im2 = z_im * z_im;
    int i;
    #pragma unroll 8
    for (i = 0; i < MAX_ITERATIONS; ++i) {
        if (z_re2 + z_im2 > 4.f)
            break;
        z_im = __fmaf_rn(2.f, z_re * z_im, c_im);
        z_re = __fadd_rn(z_re2 - z_im2, c_re);
        z_re2 = z_re * z_re;
        z_im2 = z_im * z_im;
    }
    return i;
}

__global__ void mandelbrotKernel(unsigned char* output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    
    for (int i = idx; i < WIDTH * HEIGHT; i += stride) {
        int x = i % WIDTH;
        int y = i / WIDTH;
        float c_re = (x - WIDTH/2.f) * 4.f/WIDTH;
        float c_im = (y - HEIGHT/2.f) * 4.f/HEIGHT;
        int value = mandelbrot(c_re, c_im);
        
        // Map the iteration count to a color
        unsigned char color = (unsigned char)((value * 255) / MAX_ITERATIONS);
        output[i] = color;
    }
}

int main() {
    unsigned char *d_output, *h_output;
    size_t size = WIDTH * HEIGHT * sizeof(unsigned char);
    h_output = (unsigned char*)malloc(size);
    cudaMalloc(&d_output, size);

    int blockSize = BLOCK_SIZE;
    int numBlocks = (WIDTH * HEIGHT + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    mandelbrotKernel<<<numBlocks, blockSize>>>(d_output);
    cudaEventRecord(stop);

    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Kernel execution time: %f ms\n", milliseconds);

    // Save the image
    stbi_write_png("mandelbrot.png", WIDTH, HEIGHT, 1, h_output, WIDTH);
    printf("Image saved as mandelbrot.png\n");

    cudaFree(d_output);
    free(h_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}