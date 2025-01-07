#include "cuda_utils.h"
#include "kernels.cuh"
#include "sensors.cuh"
#include "surfaceData.cuh"

string latent_heat_flux_function(Products products, Tensor tensors)
{
    int64_t initial_time, final_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    cudaEventRecord(start);
    float pos1 = 1;
    float neg1 = -1;
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add, (void *)&pos1, products.net_radiation_d, (void *)&neg1, products.soil_heat_d, products.tensor_aux1_d, tensors.stream));
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add, (void *)&pos1, products.tensor_aux1_d, (void *)&neg1, products.sensible_heat_flux_d, products.latent_heat_flux_d, tensors.stream));
    cudaEventRecord(stop);

    float cuda_time = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuda_time, start, stop);
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    return "KERNELS,LATENT_HEAT_FLUX," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string net_radiation_24h_function(Products products, Station station, MTL mtl, Tensor tensors)
{
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

    int FL = 110;
    float pos1 = 1;
    float neg1 = -1;
    float negFLRsRa = -(FL * Rs24h / Ra24h);
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add, (void *)&pos1, products.only1_d, (void *)&neg1, products.albedo_d, products.tensor_aux1_d, tensors.stream));
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add, (void *)&Rs24h, products.tensor_aux1_d, (void *)&negFLRsRa, products.only1_d, products.net_radiation_24h_d, tensors.stream));
    cudaEventRecord(stop);

    float cuda_time = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuda_time, start, stop);
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    return "KERNELS,NET_RADIATION_24H," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string evapotranspiration_24h_function(Products products, Station station, Tensor tensors)
{
    int64_t initial_time, final_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    cudaEventRecord(start);
    float pos2501 = 2.501;
    float neg00236 = -0.0236;
    float pos86400 = 86400;
    float pow10 = pow(10, 6);
    float neg1 = -1;
    float pos1 = 1;
    float neg27315 = -273.15;

    // (86400 / ((2.501 - 0.0236 * temperature_celcius) * pow(10, 6)))
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add, (void *)&pos1, products.surface_temperature_d, (void *)&neg27315, products.only1_d, products.tensor_aux1_d, tensors.stream));
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add, (void *)&pos2501, products.only1_d, (void *)&neg00236, products.tensor_aux1_d, products.tensor_aux1_d, tensors.stream));
    HANDLE_CUTENSOR_ERROR(cutensorPermute(tensors.handle, tensors.tensor_plan_permute_id, (void *)&pow10, products.tensor_aux1_d, products.tensor_aux1_d, tensors.stream));
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_div, (void *)&pos86400, products.only1_d, (void *)&pos1, products.tensor_aux1_d, products.tensor_aux1_d, tensors.stream));

    // (latent_heat_flux_d[pos] / (net_radiation_d[pos] - soil_heat_d[pos]))
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add, (void *)&pos1, products.net_radiation_d, (void *)&neg1, products.soil_heat_d, products.evapotranspiration_24h_d, tensors.stream));
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_div, (void *)&pos1, products.latent_heat_flux_d, (void *)&pos1, products.evapotranspiration_24h_d, products.evapotranspiration_24h_d, tensors.stream));

    // evapotranspiration_24h_d[pos] = (86400 / ((2.501 - 0.0236 * temperature_celcius) * pow(10, 6))) * (latent_heat_flux_d[pos] / (net_radiation_d[pos] - soil_heat_d[pos])) * net_radiation_24h_d[pos];
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_mult, (void *)&pos1, products.tensor_aux1_d, (void *)&pos1, products.evapotranspiration_24h_d, products.evapotranspiration_24h_d, tensors.stream));
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_mult, (void *)&pos1, products.evapotranspiration_24h_d, (void *)&pos1, products.net_radiation_24h_d, products.evapotranspiration_24h_d, tensors.stream));
    cudaEventRecord(stop);

    float cuda_time = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuda_time, start, stop);
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    return "KERNELS,EVAPOTRANSPIRATION_24H," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::compute_H_ET(Products products, Station station, MTL mtl, Tensor tensors)
{
    string result = "";
    int64_t initial_time, final_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    cudaEventRecord(start);
    result += latent_heat_flux_function(products, tensors);
    result += net_radiation_24h_function(products, station, mtl, tensors);
    result += evapotranspiration_24h_function(products, station, tensors);
    cudaEventRecord(stop);

    float cuda_time = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuda_time, start, stop);
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    result += "KERNELS,P4_FINAL_PROD," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
    return result;
};
