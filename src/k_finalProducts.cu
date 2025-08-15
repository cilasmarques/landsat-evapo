#include "kernels.cuh"

__global__ void latent_heat_flux_kernel(double *net_radiation_d, double *soil_heat_d, double *sensible_heat_flux_d, double *latent_heat_flux_d)
{
    unsigned int pos = threadIdx.x + blockIdx.x * blockDim.x;

    if (pos < width_d * height_d) {
        latent_heat_flux_d[pos] = net_radiation_d[pos] - soil_heat_d[pos] - sensible_heat_flux_d[pos];
    }
}

__global__ void net_radiation_24h_kernel(double *albedo_d, float Rs24h, float Ra24h, double *net_radiation_24h_d)
{
    unsigned int pos = threadIdx.x + blockIdx.x * blockDim.x;
    float FL = 110.0f;
    if (pos < width_d * height_d) {
        net_radiation_24h_d[pos] = (1.0f - albedo_d[pos]) * Rs24h - FL * Rs24h / Ra24h;
    }
}

__global__ void evapotranspiration_24h_kernel(double *surface_temperature_d, double *latent_heat_flux_d, double *net_radiation_d, double *soil_heat_d, double *net_radiation_24h_d, double *evapotranspiration_24h_d)
{
    unsigned int pos = threadIdx.x + blockIdx.x * blockDim.x;

    if (pos < width_d * height_d) {
        float temperature_celcius = surface_temperature_d[pos] - 273.15f;
        evapotranspiration_24h_d[pos] = (86400.0f / ((2.501f - 0.0236f * temperature_celcius) * powf(10.0f, 6.0f))) * 
                                        (latent_heat_flux_d[pos] / (net_radiation_d[pos] - soil_heat_d[pos])) * 
                                        net_radiation_24h_d[pos];
    }
}
