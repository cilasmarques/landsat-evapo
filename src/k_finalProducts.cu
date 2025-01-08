#include "kernels.cuh"

__global__ void latent_heat_flux_kernel(half *net_radiation_d, half *soil_heat_d, half *sensible_heat_flux_d, half *latent_heat_flux_d)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < width_d * height_d) {
        unsigned int row = idx / width_d;
        unsigned int col = idx % width_d;
        unsigned int pos = row * width_d + col;
        latent_heat_flux_d[pos] = net_radiation_d[pos] - soil_heat_d[pos] - sensible_heat_flux_d[pos];
    }
}

__global__ void net_radiation_24h_kernel(half *albedo_d, float Rs24h, float Ra24h, half *net_radiation_24h_d)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    int FL = 110;
    if (idx < width_d * height_d) {
        unsigned int row = idx / width_d;
        unsigned int col = idx % width_d;
        unsigned int pos = row * width_d + col;
        net_radiation_24h_d[pos] = __float2half((1.0f - __half2float(albedo_d[pos])) * Rs24h - FL * Rs24h / Ra24h);
    }
}

__global__ void evapotranspiration_24h_kernel(half *surface_temperature_d, half *latent_heat_flux_d, half *net_radiation_d, half *soil_heat_d, half *net_radiation_24h_d, half *evapotranspiration_24h_d)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < width_d * height_d) {
        unsigned int row = idx / width_d;
        unsigned int col = idx % width_d;
        unsigned int pos = row * width_d + col;
        float temperature_celcius = __half2float(surface_temperature_d[pos]) - 273.15f;
        evapotranspiration_24h_d[pos] = __float2half((86400.0f / ((2.501f - 0.0236f * temperature_celcius) * pow(10, 6))) * (__half2float(latent_heat_flux_d[pos]) / (__half2float(net_radiation_d[pos]) - __half2float(soil_heat_d[pos]))) * __half2float(net_radiation_24h_d[pos]));
    }
}
