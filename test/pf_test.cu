#include <limits.h>
#include <math.h>
#include "gtest/gtest.h"
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#include "pf/pf_support.cuh"

namespace {

    __global__ void inner_product_kernel(float* vec1, float* vec2, int length, float* result) {
        *result = inner_product(vec1, vec2, length);
    }

    float gpu_inner_product(float* vec1, float* vec2, int length) {
        float *gpu_vec1, *gpu_vec2, *result;
        int size = sizeof(float) * length;
        printf("therrkkk");
        // cudaMalloc((void**) &gpu_vec1, sizeof(float) * 2);
        // cudaMalloc((void**) &gpu_vec2, size);
        // cudaMalloc((void**) &result, sizeof(float));
        // cudaMemcpy(gpu_vec1, vec1, size, cudaMemcpyHostToDevice);
        // cudaMemcpy(gpu_vec2, vec2, size, cudaMemcpyHostToDevice);
        // inner_product_kernel<<<1, 1>>>(gpu_vec1, gpu_vec2, length, result);
        return *result;
    }

    float* estimate_measurement(float* vec) {
        float* result = alloc_float(2);
        memcpy(result, vec, 2 * sizeof(float));
        return result;
    }
    float* step_process(float* vec) {
        float* result = alloc_float(2);
        memcpy(result, vec, 2 * sizeof(float));
        return result;
    }

    bool eq(float* vec1, float* vec2, int length) {
        bool acc = true;
        for (int i = 0; i < length; i++) {
            acc &= (vec1[i] == vec2[i]);
        }
        return acc;
    }

    TEST(GPUInnerProductTest, NormSquared) {
        float vec1[2] = {1.0f, 2.0f};
        float vec2[2] = {1.0f, 2.0f};
        EXPECT_EQ(5, gpu_inner_product(vec1, vec2, 2));
    }

    // TEST(InnerProductTest, Any) {
    //     float vec1[2] = {3.0f, 5.0f};
    //     float vec2[2] = {1.0f, 2.0f};
    //     EXPECT_EQ(13, inner_product(vec1, vec2, 2));
    // }

    // TEST(InnerProductTest, Orthogonal) {
    //     float vec1[2] = {-1.0f, 1.0f};
    //     float vec2[2] = {1.0f, 1.0f};
    //     EXPECT_EQ(0, inner_product(vec1, vec2, 2));
    // }


    // TEST(VecSubtractTest, Any) {
    //     float vec1[2] = {-1.0f, 1.0f};
    //     float vec2[2] = {1.0f, 1.0f};
    //     float result[2] = {-2.0, 0.0};
    //     EXPECT_TRUE(eq(vec_subtract(vec1, vec2, 2), result, 2));
    // }


    // TEST(MatVecMul, Identity) {
    //     float vec[] = {-1.0f, 1.0f};
    //     float mat[] = {1.0f, 0.0f, 0.0f, 1.0f};
    //     EXPECT_TRUE(eq(mat_vec_mul(mat, vec, 2, 2), vec, 2));
    // }

    // TEST(MatVecMul, Ones) {
    //     float vec[] = {1.0f, 1.0f};
    //     float mat[] = {1.0f, 1.0f, 1.0f, 1.0f};
    //     float result[] = {2.0f, 2.0f};
    //     EXPECT_TRUE(eq(mat_vec_mul(mat, vec, 2, 2), result, 2));
    // }


    // TEST(CalcNormSquaredIn, Ones) {
    //     float vec[] = {1.0f, 1.0f};
    //     float mat[] = {1.0f, 1.0f, 1.0f, 1.0f};
    //     EXPECT_EQ(calc_norm_squared_in(vec, mat, 2), 4);
    // }


    // TEST(CalcUnnormalizedImportanceWeight, White) {
    //     float initial_state[] = {1.0f, 1.0f};
    //     float cov[] = {1.0f, 0.0f, 0.0f, 1.0f};
    //     systemModel model = {2, 2, initial_state, cov, cov, cov, cov, estimate_measurement, step_process};
    //     float current_estimate[] = {0.0f, 0.0f};
    //     EXPECT_FLOAT_EQ(calc_unnormalized_importance_weight(model, current_estimate, initial_state), 1.0 / M_E);

    //     float current_estimate_exact[] = {1.0f, 1.0f};
    //     EXPECT_FLOAT_EQ(calc_unnormalized_importance_weight(model, current_estimate_exact, initial_state), 1);
    // }


    // TEST(VecMutateDivide, One) {
    //     float vec[] = {2.0f, 3.0f};
    //     vec_mutate_divide(vec, 1.0, 2);
    //     EXPECT_TRUE(eq(vec, vec, 2));
    // }


    // TEST(VecMutateDivide, Two) {
    //     float vec[] = {2.0f, 3.0f};
    //     vec_mutate_divide(vec, 2.0, 2);
    //     float result[] = {1.0f, 1.5f};
    //     EXPECT_TRUE(eq(vec, result, 2));
    // }


    // TEST(Sum, Any) {
    //     float vec[] = {2.0f, 3.0f};
    //     EXPECT_FLOAT_EQ(sum(vec, 2), 5);
    // }


    // TEST(Cumsum, Any) {
    //     float vec[] = {2.0f, 3.0f, 5.0f};
    //     float result[] = {2.0f, 5.0f, 10.0f};
    //     EXPECT_TRUE(eq(cumsum(vec, 3), result, 3));
    // }


    // TEST(VecMutateAdd, Any) {
    //     float vec1[] = {2.0f, 3.0f, 5.0f};
    //     float vec2[] = {1.0f, 2.0f, 3.0f};
    //     vec_mutate_add(vec1, vec2, 3);
    //     float result[] = {3.0f, 5.0f, 8.0f};
    //     EXPECT_TRUE(eq(vec1, result, 3));
    // }


    // TEST(AddNoise, Any) {
    //     curandState* state;
    //     curand_init(1234, 0, 0, state);
    //     float vec[] = {2.0f, 3.0f, 5.0f};
    //     float noise_cov_sqrt[] = {sqrt(0.1f), 0.0f, 0.0f,
    //                               0.0f, sqrt(0.1f), 0.0f,
    //                               0.0f, 0.0f, sqrt(0.1f)};
    //     add_noise(vec, noise_cov_sqrt, 3, state);
    //     EXPECT_LT(vec[0], 3.0);
    //     EXPECT_GT(vec[0], 1.0);
    //     EXPECT_LT(vec[1], 4.0);
    //     EXPECT_GT(vec[1], 2.0);
    //     EXPECT_LT(vec[2], 6.0);
    //     EXPECT_GT(vec[2], 4.0);
    // }


    // TEST(Resample, Any) {
    //     curandState* state;
    //     curand_init(1234, 0, 0, state);
    //     float* particles = alloc_float(4);
    //     particles[0] = 2.0f;
    //     particles[1] = 3.0f;
    //     particles[2] = 5.0f;
    //     particles[3] = 1.0f;
    //     float weights[] = {0.3f, 0.7f};
    //     float* resampled_particles = resample_particles(particles, weights, 2, 2, state);
    //     EXPECT_TRUE(resampled_particles != particles);
    // }

}
