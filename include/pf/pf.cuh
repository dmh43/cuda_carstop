#include "./pf_support.h"

__global__ void pf(float* estimates, float* measurements, systemModel model, int num_samples, int num_particles, curandState* states);
