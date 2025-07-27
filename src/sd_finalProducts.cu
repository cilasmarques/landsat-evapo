#include "cuda_utils.h"
#include "kernels.cuh"
#include "sensors.cuh"
#include "surfaceData.cuh"

string latent_heat_flux_function(Products products)
{
    int64_t initial_time, final_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    cudaEventRecord(start);
    latent_heat_flux_kernel<<<blocks_n, threads_n>>>(products.net_radiation_d, products.soil_heat_d, products.sensible_heat_flux_d, products.latent_heat_flux_d);
    cudaEventRecord(stop);

    float cuda_time = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuda_time, start, stop);
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    return "KERNELS,LATENT_HEAT_FLUX," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string net_radiation_24h_function(Products products, Station station, MTL mtl)
{
    int64_t initial_time, final_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    cudaEventRecord(start);
    float dr = (1.0f / mtl.distance_earth_sun) * (1.0f / mtl.distance_earth_sun);
    float sigma = 0.409f * sinf(((2.0f * PI / 365.0f) * mtl.julian_day) - 1.39f);
    float phi = (PI / 180.0f) * station.latitude;
    float omegas = acosf(-tanf(phi) * tanf(sigma));
    float Ra24h = (((24.0f * 60.0f / PI) * GSC * dr) * (omegas * sinf(phi) * sinf(sigma) + cosf(phi) * cosf(sigma) * sinf(omegas))) * (1000000.0f / 86400.0f);
    float Rs24h = station.INTERNALIZATION_FACTOR * sqrtf(station.v7_max - station.v7_min) * Ra24h;
    net_radiation_24h_kernel<<<blocks_n, threads_n>>>(products.albedo_d, Rs24h, Ra24h, products.net_radiation_24h_d);
    cudaEventRecord(stop);

    float cuda_time = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuda_time, start, stop);
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    return "KERNELS,NET_RADIATION_24H," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string evapotranspiration_24h_function(Products products, Station station)
{
    int64_t initial_time, final_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    cudaEventRecord(start);
    evapotranspiration_24h_kernel<<<blocks_n, threads_n>>>(products.surface_temperature_d, products.latent_heat_flux_d, products.net_radiation_d, products.soil_heat_d, products.net_radiation_24h_d, products.evapotranspiration_24h_d);
    cudaEventRecord(stop);

    float cuda_time = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuda_time, start, stop);
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    return "KERNELS,EVAPOTRANSPIRATION_24H," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::compute_H_ET(Products products, Station station, MTL mtl)
{
    string result = "";
    int64_t initial_time, final_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    cudaEventRecord(start);
    result += latent_heat_flux_function(products);
    result += net_radiation_24h_function(products, station, mtl);
    result += evapotranspiration_24h_function(products, station);
    cudaEventRecord(stop);

    float cuda_time = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuda_time, start, stop);
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    result += "KERNELS,P4_FINAL_PROD," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
    return result;
};
