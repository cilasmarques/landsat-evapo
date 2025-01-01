#include "sensors.cuh"
#include "surfaceData.cuh"
#include "kernels.cuh"
#include "cuda_utils.h"

string sensible_heat_flux_function(Products products)
{
    int64_t initial_time, final_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    cudaEventRecord(start);
    sensible_heat_flux_kernel<<<products.blocks_num, products.threads_num>>>(products.d_hotCandidates, products.d_coldCandidates, products.surface_temperature_d,
                                                           products.rah_d, products.net_radiation_d, products.soil_heat_d, products.sensible_heat_flux_d, products.width_band, products.height_band);

    cudaEventRecord(stop);

    float cuda_time = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuda_time, start, stop);
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    return "KERNELS,SENSIBLE_HEAT_FLUX," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string latent_heat_flux_function(Products products)
{
    int64_t initial_time, final_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    cudaEventRecord(start);
    latent_heat_flux_kernel<<<products.blocks_num, products.threads_num>>>(products.net_radiation_d, products.soil_heat_d, products.sensible_heat_flux_d, products.latent_heat_flux_d, products.width_band, products.height_band);
    cudaEventRecord(stop);

    float cuda_time = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuda_time, start, stop);
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    return "KERNELS,LATENT_HEAT_FLUX," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string net_radiation_24h_function(Products products, float Ra24h, float Rs24h)
{
    int64_t initial_time, final_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    cudaEventRecord(start);
    net_radiation_24h_kernel<<<products.blocks_num, products.threads_num>>>(products.albedo_d, Rs24h, Ra24h, products.net_radiation_24h_d, products.width_band, products.height_band);
    cudaEventRecord(stop);

    float cuda_time = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuda_time, start, stop);
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    return "KERNELS,NET_RADIATION_24H," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string evapotranspiration_fraction_fuction(Products products)
{
    int64_t initial_time, final_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    cudaEventRecord(start);
    evapotranspiration_fraction_kernel<<<products.blocks_num, products.threads_num>>>(products.net_radiation_d, products.soil_heat_d, products.latent_heat_flux_d, products.evapotranspiration_fraction_d, products.width_band, products.height_band);
    cudaEventRecord(stop);

    float cuda_time = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuda_time, start, stop);
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    return "KERNELS,EVAPOTRANSPIRATION_FRACTION," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string sensible_heat_flux_24h_fuction(Products products)
{
    int64_t initial_time, final_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    cudaEventRecord(start);
    sensible_heat_flux_24h_kernel<<<products.blocks_num, products.threads_num>>>(products.net_radiation_24h_d, products.evapotranspiration_fraction_d, products.sensible_heat_flux_24h_d, products.width_band, products.height_band);
    cudaEventRecord(stop);

    float cuda_time = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuda_time, start, stop);
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    return "KERNELS,SENSIBLE_HEAT_FLUX_24H," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string latent_heat_flux_24h_function(Products products)
{
    int64_t initial_time, final_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    cudaEventRecord(start);
    latent_heat_flux_24h_kernel<<<products.blocks_num, products.threads_num>>>(products.net_radiation_24h_d, products.evapotranspiration_fraction_d, products.latent_heat_flux_24h_d, products.width_band, products.height_band);
    cudaEventRecord(stop);

    float cuda_time = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuda_time, start, stop);
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    return "KERNELS,LATENT_HEAT_FLUX_24H," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string evapotranspiration_24h_function(Products products, Station station)
{
    int64_t initial_time, final_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    cudaEventRecord(start);
    evapotranspiration_24h_kernel<<<products.blocks_num, products.threads_num>>>(products.latent_heat_flux_24h_d, products.evapotranspiration_24h_d, station.v7_max, station.v7_min, products.width_band, products.height_band);
    cudaEventRecord(stop);

    float cuda_time = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuda_time, start, stop);
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    return "KERNELS,EVAPOTRANSPIRATION_24H," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string evapotranspiration_function(Products products)
{
    int64_t initial_time, final_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    cudaEventRecord(start);
    evapotranspiration_kernel<<<products.blocks_num, products.threads_num>>>(products.net_radiation_24h_d, products.evapotranspiration_fraction_d, products.evapotranspiration_d, products.width_band, products.height_band);
    cudaEventRecord(stop);

    float cuda_time = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuda_time, start, stop);
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    return "KERNELS,EVAPOTRANSPIRATION," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
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
    float dr = (1 / mtl.distance_earth_sun) * (1 / mtl.distance_earth_sun);
    float sigma = 0.409 * sin(((2 * PI / 365) * mtl.julian_day) - 1.39);
    float phi = (PI / 180) * station.latitude;
    float omegas = acos(-tan(phi) * tan(sigma));
    float Ra24h = (((24 * 60 / PI) * GSC * dr) * (omegas * sin(phi) * sin(sigma) + cos(phi) * cos(sigma) * sin(omegas))) * (1000000 / 86400.0);
    float Rs24h = station.INTERNALIZATION_FACTOR * sqrt(station.v7_max - station.v7_min) * Ra24h;

    result += sensible_heat_flux_function(products);
    result += latent_heat_flux_function(products);
    result += net_radiation_24h_function(products, Ra24h, Rs24h);
    result += evapotranspiration_fraction_fuction(products);
    result += sensible_heat_flux_24h_fuction(products);
    result += latent_heat_flux_24h_function(products);
    result += evapotranspiration_24h_function(products, station);
    result += evapotranspiration_function(products);
    cudaEventRecord(stop);

    float cuda_time = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuda_time, start, stop);
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    result += "KERNELS,P4_FINAL_PROD," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
    return result;
};
