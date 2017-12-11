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

auto float inner_product(float* vec1, float* vec2, int length);

auto float* vec_subtract(float* vec1, float* vec2, int length);

auto float* mat_vec_mul(float* mat, float* vec, int input_length, int result_length);

auto float calc_norm_squared_in(float* vec, float* mat, int vec_length);

auto void update_particle(systemModel model, float* current_state_estimate, float* current_measurement);

auto void vec_mutate_divide(float* vec, float divisor, int length);

auto float sum(float* vec, int length);

auto float* cumsum(float* vec, int length);

auto void vec_mutate_add(float* vec1, float* vec2, int length);

auto float normal_rand();

auto float* random_normal_vector(int length, curandState* state);

auto void add_noise(float* vec, float* noise_covariance_sqrt, int length, curandState* state);

auto float calc_unnormalized_importance_weight(systemModel model, float* current_state_estimate, float* current_measurement);

auto void update_importance_weights(float* weights, systemModel model, float* current_measurement, float* particles, int num_particles);

auto void update_estimates(float* estimate, float* weights, float* particles, int num_particles, int num_state_variables);

auto float* resample_particles(float* particles, float* weights, int num_particles, int num_state_variables, curandState* state);

auto void predict_particles_step(systemModel model, float* particles, int num_particles, curandState* states);

auto float* initialize_particles(systemModel model, int num_particles, curandState* state);

auto void pf(float* estimates, float* measurements, systemModel model, int num_samples, int num_particles, curandState* states);
