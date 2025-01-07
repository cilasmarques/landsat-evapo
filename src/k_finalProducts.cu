#include "kernels.cuh"

__global__ void latent_heat_flux_kernel(float *net_radiation_d, float *soil_heat_d, float *sensible_heat_flux_d, float *latent_heat_flux_d)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < width_d * height_d) {
        unsigned int row = idx / width_d;
        unsigned int col = idx % width_d;
        unsigned int pos = row * width_d + col;
        latent_heat_flux_d[pos] = net_radiation_d[pos] - soil_heat_d[pos] - sensible_heat_flux_d[pos];
    }
}

__global__ void net_radiation_24h_kernel(float *albedo_d, float Rs24h, float Ra24h, float *net_radiation_24h_d)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    int FL = 110;
    if (idx < width_d * height_d) {
        unsigned int row = idx / width_d;
        unsigned int col = idx % width_d;
        unsigned int pos = row * width_d + col;
        net_radiation_24h_d[pos] = (1 - albedo_d[pos]) * Rs24h - FL * Rs24h / Ra24h;
    }
}

__global__ void evapotranspiration_24h_kernel(float *surface_temperature_d, float *latent_heat_flux_d, float *net_radiation_d, float *soil_heat_d, float *net_radiation_24h_d, float *evapotranspiration_24h_d)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < width_d * height_d) {
        unsigned int row = idx / width_d;
        unsigned int col = idx % width_d;
        unsigned int pos = row * width_d + col;
        float temperature_celcius = surface_temperature_d[pos] - 273.15;
        evapotranspiration_24h_d[pos] = (86400 / ((2.501 - 0.0236 * temperature_celcius) * pow(10, 6))) * (latent_heat_flux_d[pos] / (net_radiation_d[pos] - soil_heat_d[pos])) * net_radiation_24h_d[pos];
    }
}
