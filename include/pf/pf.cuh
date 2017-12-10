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

float inner_product(float* vec1, float* vec2, int length);

float* vec_subtract(float* vec1, float* vec2, int length);

float* mat_vec_mul(float* mat, float* vec, int input_length, int result_length);

float calc_norm_squared_in(float* vec, float* mat, int vec_length);

void update_particle(systemModel model, float* current_state_estimate, float* current_measurement);

float* pf(float* measurements, systemModel model, int num_samples, int num_particles);

void vec_mutate_divide(float* vec, float divisor, int length);

float sum(float* vec, int length);

float* cumsum(float* vec, int length);

void vec_mutate_add(float* vec1, float* vec2, int length);

float normal_rand();

float* random_normal_vector(int length);

void add_noise(float* vec, float* noise_covariance_sqrt, int length);

float calc_unnormalized_importance_weight(systemModel model, float* current_state_estimate, float* current_measurement);

void update_importance_weights(float* weights, systemModel model, float* current_measurement, float* particles, int num_particles);

void update_estimates(float* estimate, float* weights, float* particles, int num_particles, int num_state_variables);

float* resample_particles(float* particles, float* weights, int num_particles, int num_state_variables);

void predict_particles_step(systemModel model, float* particles, int num_particles);

float* initialize_particles(systemModel model, int num_particles);
