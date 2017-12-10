#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "pf/pf.cuh"

__device__ float inner_product(float* vec1, float* vec2, int length) {
    float acc = 0;
    for (int vector_index = 0; vector_index < length; vector_index++) {
        acc += vec1[vector_index] * vec2[vector_index];
    }
    return acc;
}

__device__ float* vec_subtract(float* vec1, float* vec2, int length) {
    float* result = alloc_float(length);
    for (int vector_index = 0; vector_index < length; vector_index++) {
        result[vector_index] = vec1[vector_index] - vec2[vector_index];
    }
    return result;
}

__global__ void vec_mutate_divide(float* vec, float divisor, int length) {
    for (int vector_index = 0; vector_index < length; vector_index++) {
        vec[vector_index] = vec[vector_index] / divisor;
    }
}

__device__ float sum(float* vec, int length) {
    float acc = 0;
    for (int vector_index = 0; vector_index < length; vector_index++) {
        acc += vec[vector_index];
    }
    return acc;
}

__device__ float* cumsum(float* vec, int length) {
    float* result = alloc_float(length);
    for (int vector_index = 0; vector_index < length; vector_index++) {
        if (vector_index == 0) {
            result[vector_index] = vec[vector_index];
        } else {
            result[vector_index] = result[vector_index - 1] + vec[vector_index];
        }
    }
    return result;
}

__device__ float* mat_vec_mul(float* mat, float* vec, int input_length, int result_length) {
    float* result = alloc_float(result_length);
    for (int matrix_row_index = 0; matrix_row_index < result_length; matrix_row_index++) {
        float* matrix_row = &mat[matrix_row_index * input_length];
        result[matrix_row_index] = inner_product(matrix_row, vec, input_length);
    }
    return result;
}

__device__ float calc_norm_squared_in(float* vec, float* mat, int vec_length) {
    float* first_product_result = mat_vec_mul(mat, vec, vec_length, vec_length);
    float result = inner_product(first_product_result, vec, vec_length);
    free(first_product_result);
    return result;
}

__global__ void vec_mutate_add(float* vec1, float* vec2, int length) {
    for (int vector_index = 0; vector_index < length; vector_index++) {
        vec1[vector_index] += vec2[vector_index];
    }
}

__device__ float normal_rand() {
    float u = ((float) rand() / (RAND_MAX)) * 2 - 1;
    float v = ((float) rand() / (RAND_MAX)) * 2 - 1;
    float r = u * u + v * v;
    if (r == 0 || r > 1) return normal_rand();
    float c = sqrt(-2 * log(r) / r);
    return u * c;
}

__device__ float* random_normal_vector(int length) {
    float* result = alloc_float(length);
    for (int vector_index = 0; vector_index < length; vector_index++) {
        result[vector_index] = normal_rand();
    }
    return result;
}

__global__ void add_noise(float* vec, float* noise_covariance_sqrt, int length) {
    float* random = random_normal_vector(length);
    float* first_result = mat_vec_mul(noise_covariance_sqrt, random, length, length);
    vec_mutate_add(vec, first_result, length);
    free(first_result);
    free(random);
}

__device__ float calc_unnormalized_importance_weight(systemModel model, float* current_state_estimate, float* current_measurement) {
    float* predicted_measurement = model.estimate_measurement(current_state_estimate);
    float* error = vec_subtract(current_measurement, predicted_measurement, model.num_measurement_variables);
    float unnormalized_weight = exp(-0.5 * calc_norm_squared_in(error,
                                                                model.measurement_noise_covariance_inv,
                                                                model.num_measurement_variables));
    free(error);
    free(predicted_measurement);
    return unnormalized_weight;
}

__global__ void update_importance_weights(float* weights, systemModel model, float* current_measurement, float* particles, int num_particles) {
    for (int weight_index = 0; weight_index < num_particles; weight_index++) {
        float* current_state_estimate = &particles[weight_index * model.num_state_variables];
        weights[weight_index] = calc_unnormalized_importance_weight(model,
                                                                    current_state_estimate,
                                                                    current_measurement);
    }
    vec_mutate_divide(weights, sum(weights, num_particles), num_particles);
}

__global__ void update_estimates(float* estimate, float* weights, float* particles, int num_particles, int num_state_variables) {
    for (int variable_index = 0; variable_index < num_state_variables; variable_index++) {
        estimate[variable_index] = 0;
        for (int weight_index = 0; weight_index < num_particles; weight_index++) {
            estimate[variable_index] += weights[weight_index] * particles[weight_index * num_state_variables + variable_index];
        }
    }
}

__device__ float* resample_particles(float* particles, float* weights, int num_particles, int num_state_variables) {
    float* resampled_particles = alloc_float(num_particles * num_state_variables);
    float* reference = cumsum(weights, num_particles);
    for (int weight_index = 0; weight_index < num_particles; weight_index++) {
        float uniform_sample = (weight_index + rand_f()) / num_particles;
        int sum_index = 0;
        while (reference[sum_index] < uniform_sample) {
            sum_index++;
        }
        memcpy(&resampled_particles[weight_index * num_state_variables],
               &particles[sum_index * num_state_variables],
               num_state_variables * sizeof(float));
    }
    free(particles);
    free(reference);
    return resampled_particles;
}

__global__ void predict_particles_step(systemModel model, float* particles, int num_particles) {
    for (int particle_index = 0; particle_index < num_particles * model.num_state_variables; particle_index += model.num_state_variables) {
        float* particle = &particles[particle_index];
        float* process_estimate = model.step_process(particle);
        add_noise(process_estimate, model.process_noise_covariance_sqrt, model.num_state_variables);
        memcpy(particle, process_estimate, model.num_state_variables * sizeof(float));
        free(process_estimate);
    }
}

__device__ float* initialize_particles(systemModel model, int num_particles) {
    float* particles = alloc_float(num_particles * model.num_state_variables);
    for (int particle_index = 0; particle_index < num_particles * model.num_state_variables; particle_index += model.num_state_variables) {
        float* particle = &particles[particle_index];
        memcpy(particle, model.initial_state, model.num_state_variables * sizeof(float));
        add_noise(particle, model.initial_state_covariance_sqrt, model.num_state_variables);
    }
    return particles;
}

__device__ float* pf(float* measurements, systemModel model, int num_samples, int num_particles) {
    float* particles = initialize_particles(model, num_particles);
    float* weights = alloc_float(num_particles);
    float* estimates = alloc_float(num_samples * model.num_state_variables);
    for (int sample_index = 0; sample_index < num_samples; sample_index++) {
        float* current_measurement = &measurements[sample_index * model.num_measurement_variables];
        update_importance_weights(weights, model, current_measurement, particles, num_particles);
        update_estimates(&estimates[sample_index * model.num_state_variables], weights, particles, num_particles, model.num_state_variables);
        particles = resample_particles(particles, weights, num_particles, model.num_state_variables);
        predict_particles_step(model, particles, num_particles);
    }
    free(weights);
    free(particles);
    return estimates;
}
