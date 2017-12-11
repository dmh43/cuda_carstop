#include "pf_support.cuh"

__global__ void pf(float* estimates, float* measurements, systemModel model, int num_samples, int num_particles, curandState* states);
