#include <cuda.h>
#include "../include/pf/pf_support.cuh"

void inner_product_kernel(float* vec1, float* vec2, int length, float* result) {
    *result = inner_product(vec1, vec2, length);
}

float gpu_inner_product(float* vec1, float* vec2, int length) {
    float *gpu_vec1, *gpu_vec2, *result;
    int size = sizeof(float) * length;
    cudaMalloc(&gpu_vec1, size);
    cudaMalloc(&gpu_vec2, size);
    cudaMalloc(&result, sizeof(float));
    cudaMemcpy(gpu_vec1, vec1, size, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_vec2, vec2, size, cudaMemcpyHostToDevice);
    inner_product_kernel<<<1, 1>>>(gpu_vec1, gpu_vec2, length, result);
    return *result;
}
