#define len(coll_ptr) (sizeof(coll_ptr)/sizeof(coll_ptr[0]))
#define alloc_float(num_elems) ((float*) malloc(num_elems * sizeof(float)))
#define rand_f() ((float)rand()/(float)(RAND_MAX))

typedef float* (*float_p_float_p_fptr)(float*);

typedef struct systemModel {
    int num_state_variables;
    int num_measurement_variables;
    float* initial_state;
    float* initial_state_covariance_sqrt;
    float* measurement_noise_covariance_sqrt;
    float* measurement_noise_covariance_inv;
    float* process_noise_covariance_sqrt;
    float_p_float_p_fptr estimate_measurement;
    float_p_float_p_fptr step_process;
} systemModel;

__device__ float inner_product(float* vec1, float* vec2, int length);

__device__ float* vec_subtract(float* vec1, float* vec2, int length);

__device__ float* mat_vec_mul(float* mat, float* vec, int input_length, int result_length);

__device__ float calc_norm_squared_in(float* vec, float* mat, int vec_length);

__device__ void update_particle(systemModel model, float* current_state_estimate, float* current_measurement);

__device__ float* pf(float* measurements, systemModel model, int num_samples, int num_particles);

__device__ void vec_mutate_divide(float* vec, float divisor, int length);

__device__ float sum(float* vec, int length);

__device__ float* cumsum(float* vec, int length);

__device__ void vec_mutate_add(float* vec1, float* vec2, int length);

__device__ float normal_rand();

__device__ float* random_normal_vector(int length);

__device__ void add_noise(float* vec, float* noise_covariance_sqrt, int length);

__device__ float calc_unnormalized_importance_weight(systemModel model, float* current_state_estimate, float* current_measurement);

__device__ void update_importance_weights(float* weights, systemModel model, float* current_measurement, float* particles, int num_particles);

__device__ void update_estimates(float* estimate, float* weights, float* particles, int num_particles, int num_state_variables);

__device__ float* resample_particles(float* particles, float* weights, int num_particles, int num_state_variables);

__device__ void predict_particles_step(systemModel model, float* particles, int num_particles);

__device__ float* initialize_particles(systemModel model, int num_particles);
