#include "constants.h"
#include "cuda_utils.h"
#include "kernels.cuh"
#include "sensors.cuh"
#include "surfaceData.cuh"
#include "tensor.cuh"

string radiance_function(Products products, MTL mtl, Tensor tensors)
{
    int64_t initial_time, final_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    cudaEventRecord(start, 0);
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add, (void *)&mtl.rad_mult[PARAM_BAND_BLUE_INDEX], products.band_blue_d, (void *)&mtl.rad_add[PARAM_BAND_BLUE_INDEX], products.only1_d, products.radiance_blue_d, tensors.stream));
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add, (void *)&mtl.rad_mult[PARAM_BAND_GREEN_INDEX], products.band_green_d, (void *)&mtl.rad_add[PARAM_BAND_GREEN_INDEX], products.only1_d, products.radiance_green_d, tensors.stream));
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add, (void *)&mtl.rad_mult[PARAM_BAND_RED_INDEX], products.band_red_d, (void *)&mtl.rad_add[PARAM_BAND_RED_INDEX], products.only1_d, products.radiance_red_d, tensors.stream));
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add, (void *)&mtl.rad_mult[PARAM_BAND_NIR_INDEX], products.band_nir_d, (void *)&mtl.rad_add[PARAM_BAND_NIR_INDEX], products.only1_d, products.radiance_nir_d, tensors.stream));
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add, (void *)&mtl.rad_mult[PARAM_BAND_SWIR1_INDEX], products.band_swir1_d, (void *)&mtl.rad_add[PARAM_BAND_SWIR1_INDEX], products.only1_d, products.radiance_swir1_d, tensors.stream));
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add, (void *)&mtl.rad_mult[PARAM_BAND_TERMAL_INDEX], products.band_termal_d, (void *)&mtl.rad_add[PARAM_BAND_TERMAL_INDEX], products.only1_d, products.radiance_termal_d, tensors.stream));
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add, (void *)&mtl.rad_mult[PARAM_BAND_SWIR2_INDEX], products.band_swir2_d, (void *)&mtl.rad_add[PARAM_BAND_SWIR2_INDEX], products.only1_d, products.radiance_swir2_d, tensors.stream));
    NAN_kernel<<<blocks_n, threads_n>>>(products.radiance_blue_d);
    NAN_kernel<<<blocks_n, threads_n>>>(products.radiance_green_d);
    NAN_kernel<<<blocks_n, threads_n>>>(products.radiance_red_d);
    NAN_kernel<<<blocks_n, threads_n>>>(products.radiance_nir_d);
    NAN_kernel<<<blocks_n, threads_n>>>(products.radiance_swir1_d);
    NAN_kernel<<<blocks_n, threads_n>>>(products.radiance_termal_d);
    NAN_kernel<<<blocks_n, threads_n>>>(products.radiance_swir2_d);
    cudaEventRecord(stop, 0);

    float cuda_time = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuda_time, start, stop);
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    return "CUTENSOR,RADIANCE," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
}

string reflectance_function(Products products, MTL mtl, Tensor tensors)
{
    int64_t initial_time, final_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    cudaEventRecord(start);
    const float sin_sun = 1 / sin(mtl.sun_elevation * PI / 180);
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseTrinaryExecute(tensors.handle, tensors.tensor_plan_trinity_add_mult, (void *)&mtl.ref_mult[PARAM_BAND_BLUE_INDEX], products.band_blue_d, (void *)&mtl.ref_add[PARAM_BAND_BLUE_INDEX], products.only1_d, (void *)&sin_sun, products.only1_d, products.reflectance_blue_d, tensors.stream));
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseTrinaryExecute(tensors.handle, tensors.tensor_plan_trinity_add_mult, (void *)&mtl.ref_mult[PARAM_BAND_GREEN_INDEX], products.band_green_d, (void *)&mtl.ref_add[PARAM_BAND_GREEN_INDEX], products.only1_d, (void *)&sin_sun, products.only1_d, products.reflectance_green_d, tensors.stream));
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseTrinaryExecute(tensors.handle, tensors.tensor_plan_trinity_add_mult, (void *)&mtl.ref_mult[PARAM_BAND_RED_INDEX], products.band_red_d, (void *)&mtl.ref_add[PARAM_BAND_RED_INDEX], products.only1_d, (void *)&sin_sun, products.only1_d, products.reflectance_red_d, tensors.stream));
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseTrinaryExecute(tensors.handle, tensors.tensor_plan_trinity_add_mult, (void *)&mtl.ref_mult[PARAM_BAND_NIR_INDEX], products.band_nir_d, (void *)&mtl.ref_add[PARAM_BAND_NIR_INDEX], products.only1_d, (void *)&sin_sun, products.only1_d, products.reflectance_nir_d, tensors.stream));
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseTrinaryExecute(tensors.handle, tensors.tensor_plan_trinity_add_mult, (void *)&mtl.ref_mult[PARAM_BAND_SWIR1_INDEX], products.band_swir1_d, (void *)&mtl.ref_add[PARAM_BAND_SWIR1_INDEX], products.only1_d, (void *)&sin_sun, products.only1_d, products.reflectance_swir1_d, tensors.stream));
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseTrinaryExecute(tensors.handle, tensors.tensor_plan_trinity_add_mult, (void *)&mtl.ref_mult[PARAM_BAND_TERMAL_INDEX], products.band_termal_d, (void *)&mtl.ref_add[PARAM_BAND_TERMAL_INDEX], products.only1_d, (void *)&sin_sun, products.only1_d, products.reflectance_termal_d, tensors.stream));
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseTrinaryExecute(tensors.handle, tensors.tensor_plan_trinity_add_mult, (void *)&mtl.ref_mult[PARAM_BAND_SWIR2_INDEX], products.band_swir2_d, (void *)&mtl.ref_add[PARAM_BAND_SWIR2_INDEX], products.only1_d, (void *)&sin_sun, products.only1_d, products.reflectance_swir2_d, tensors.stream));
    NAN_kernel<<<blocks_n, threads_n>>>(products.reflectance_blue_d);
    NAN_kernel<<<blocks_n, threads_n>>>(products.reflectance_green_d);
    NAN_kernel<<<blocks_n, threads_n>>>(products.reflectance_red_d);
    NAN_kernel<<<blocks_n, threads_n>>>(products.reflectance_nir_d);
    NAN_kernel<<<blocks_n, threads_n>>>(products.reflectance_swir1_d);
    NAN_kernel<<<blocks_n, threads_n>>>(products.reflectance_termal_d);
    NAN_kernel<<<blocks_n, threads_n>>>(products.reflectance_swir2_d);
    cudaEventRecord(stop);

    float cuda_time = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuda_time, start, stop);
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    return "CUTENSOR,REFLECTANCE," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
}

string albedo_function(Products products, MTL mtl, Tensor tensors)
{
    int64_t initial_time, final_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    cudaEventRecord(start);
    float pos1 = 1;
    float neg003 = -0.03;
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_rcp, (void *)&pos1, products.tal_d, (void *)&pos1, products.tal_d, products.tensor_aux1_d, tensors.stream));
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add, (void *)&mtl.ref_w_coeff[PARAM_BAND_BLUE_INDEX], products.reflectance_blue_d, (void *)&mtl.ref_w_coeff[PARAM_BAND_GREEN_INDEX], products.reflectance_green_d, products.albedo_d, tensors.stream));
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add, (void *)&pos1, products.albedo_d, (void *)&mtl.ref_w_coeff[PARAM_BAND_RED_INDEX], products.reflectance_red_d, products.albedo_d, tensors.stream));
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add, (void *)&pos1, products.albedo_d, (void *)&mtl.ref_w_coeff[PARAM_BAND_NIR_INDEX], products.reflectance_nir_d, products.albedo_d, tensors.stream));
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add, (void *)&pos1, products.albedo_d, (void *)&mtl.ref_w_coeff[PARAM_BAND_SWIR1_INDEX], products.reflectance_swir1_d, products.albedo_d, tensors.stream));
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add, (void *)&pos1, products.albedo_d, (void *)&mtl.ref_w_coeff[PARAM_BAND_SWIR2_INDEX], products.reflectance_swir2_d, products.albedo_d, tensors.stream));
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseTrinaryExecute(tensors.handle, tensors.tensor_plan_trinity_add_mult, (void *)&pos1, products.albedo_d, (void *)&neg003, products.only1_d, (void *)&pos1, products.tensor_aux1_d, products.albedo_d, tensors.stream));
    NAN_kernel<<<blocks_n, threads_n>>>(products.albedo_d);
    cudaEventRecord(stop);

    float cuda_time = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuda_time, start, stop);
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    return "CUTENSOR,ALBEDO," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
}

string ndvi_function(Products products, Tensor tensors)
{
    int64_t initial_time, final_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    cudaEventRecord(start);
    float pos1 = 1;
    float neg1 = -1;
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add, (void *)&pos1, products.reflectance_nir_d, (void *)&neg1, products.reflectance_red_d, products.tensor_aux1_d, tensors.stream));
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add, (void *)&pos1, products.reflectance_nir_d, (void *)&pos1, products.reflectance_red_d, products.tensor_aux2_d, tensors.stream));
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_div, (void *)&pos1, products.tensor_aux1_d, (void *)&pos1, products.tensor_aux2_d, products.ndvi_d, tensors.stream));
    NDVI_NAN_kernel<<<blocks_n, threads_n>>>(products.ndvi_d);
    cudaEventRecord(stop);

    float cuda_time = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuda_time, start, stop);
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    return "CUTENSOR,NDVI," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string pai_function(Products products, Tensor tensors)
{
    int64_t initial_time, final_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    cudaEventRecord(start);
    float pos0 = 0;
    float pos1 = 1;
    float pos31 = 3.1;
    float pos101 = 10.1;
    float neg1 = -1;
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_sqtr_add, (void *)&pos1, products.reflectance_nir_d, (void *)&neg1, products.reflectance_red_d, products.tensor_aux1_d, tensors.stream));
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add, (void *)&pos101, products.tensor_aux1_d, (void *)&pos31, products.only1_d, products.pai_d, tensors.stream));
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_max, (void *)&pos1, products.pai_d, (void *)&pos0, products.only1_d, products.pai_d, tensors.stream));
    cudaEventRecord(stop);

    float cuda_time = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuda_time, start, stop);
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    return "CUTENSOR,PAI," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string lai_function(Products products, Tensor tensors)
{
    int64_t initial_time, final_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    cudaEventRecord(start);
    lai_kernel<<<blocks_n, threads_n>>>(products.reflectance_nir_d, products.reflectance_red_d, products.lai_d);
    cudaEventRecord(stop);

    float cuda_time = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuda_time, start, stop);
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    return "KERNELS,LAI," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string enb_emissivity_function(Products products, Tensor tensors)
{
    int64_t initial_time, final_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    cudaEventRecord(start);
    enb_kernel<<<blocks_n, threads_n>>>(products.lai_d, products.ndvi_d, products.enb_d);
    cudaEventRecord(stop);

    float cuda_time = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuda_time, start, stop);
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    return "KERNELS,ENB_EMISSIVITY," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string eo_emissivity_function(Products products, Tensor tensors)
{
    int64_t initial_time, final_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    cudaEventRecord(start);
    eo_kernel<<<blocks_n, threads_n>>>(products.lai_d, products.ndvi_d, products.eo_d);
    cudaEventRecord(stop);

    float cuda_time = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuda_time, start, stop);
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    return "KERNELS,EO_EMISSIVITY," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string ea_emissivity_function(Products products, Tensor tensors)
{
    int64_t initial_time, final_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    cudaEventRecord(start);
    float pos1 = 1;
    float pos085 = 0.85;
    float pos009 = 0.09;
    float neg1 = -1;
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_log_mul, (void *)&pos1, products.only1_d, (void *)&neg1, products.tal_d, products.tensor_aux1_d, tensors.stream));
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_log_mul, (void *)&pos009, products.only1_d, (void *)&pos1, products.tensor_aux1_d, products.tensor_aux1_d, tensors.stream));
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_exp_mul, (void *)&pos085, products.only1_d, (void *)&pos1, products.tensor_aux1_d, products.ea_d, tensors.stream));
    cudaEventRecord(stop);

    float cuda_time = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuda_time, start, stop);
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    return "CUTENSOR,EA_EMISSIVITY," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string surface_temperature_function(Products products, MTL mtl, Tensor tensors)
{
    float k1, k2;
    switch (mtl.number_sensor) {
    case 5:
        k1 = 607.76;
        k2 = 1260.56;
        break;

    case 7:
        k1 = 666.09;
        k2 = 1282.71;
        break;

    case 8:
        k1 = 774.8853;
        k2 = 1321.0789;
        break;

    default:
        cerr << "Sensor problem!";
        exit(6);
    }

    int64_t initial_time, final_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    cudaEventRecord(start);
    float pos0 = 0;
    float pos1 = 1;
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_div, (void *)&k1, products.enb_d, (void *)&pos1, products.radiance_termal_d, products.tensor_aux1_d, tensors.stream));
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add, (void *)&pos1, products.tensor_aux1_d, (void *)&pos1, products.only1_d, products.tensor_aux1_d, tensors.stream));
    HANDLE_CUTENSOR_ERROR(cutensorPermute(tensors.handle, tensors.tensor_plan_permute_log, (void *)&pos1, products.tensor_aux1_d, products.tensor_aux1_d, tensors.stream));
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_div, (void *)&k2, products.only1_d, (void *)&pos1, products.tensor_aux1_d, products.surface_temperature_d, tensors.stream));
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_max, (void *)&pos1, products.surface_temperature_d, (void *)&pos0, products.only1_d, products.surface_temperature_d, tensors.stream));
    cudaEventRecord(stop);

    float cuda_time = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuda_time, start, stop);
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    return "CUTENSOR,SURFACE_TEMPERATURE," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string short_wave_radiation_function(Products products, MTL mtl, Tensor tensors)
{
    int64_t initial_time, final_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    cudaEventRecord(start);
    float divmtl2 = 1 / (mtl.distance_earth_sun * mtl.distance_earth_sun);
    float costheta = sin(mtl.sun_elevation * PI / 180);
    float cos1367 = 1367 * costheta;
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_mult, (void *)&cos1367, products.tal_d, (void *)&divmtl2, products.only1_d, products.short_wave_radiation_d, tensors.stream));
    cudaEventRecord(stop);

    float cuda_time = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuda_time, start, stop);
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    return "CUTENSOR,SHORT_WAVE_RADIATION," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string large_wave_radiation_surface_function(Products products, Tensor tensors)
{
    int64_t initial_time, final_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    cudaEventRecord(start);
    float pos1 = 1;
    float pos567 = 5.67;
    float pos1e8 = 1e-8;
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_mult, (void *)&pos1, products.surface_temperature_d, (void *)&pos1, products.surface_temperature_d, products.tensor_aux1_d, tensors.stream));
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_mult, (void *)&pos1, products.tensor_aux1_d, (void *)&pos1, products.tensor_aux1_d, products.tensor_aux1_d, tensors.stream));
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_mult, (void *)&pos567, products.eo_d, (void *)&pos1e8, products.tensor_aux1_d, products.large_wave_radiation_surface_d, tensors.stream));
    cudaEventRecord(stop);

    float cuda_time = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuda_time, start, stop);
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    return "CUTENSOR,LARGE_WAVE_RADIATION_SURFACE," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string large_wave_radiation_atmosphere_function(Products products, Tensor tensors, float temperature)
{
    int64_t initial_time, final_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    cudaEventRecord(start);
    float pos567_1e8 = 5.67 * 1e-8;
    float temperature_kelvin = temperature;
    float temperature_kelvin_pow_4 = pow(temperature_kelvin, 4);
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_mult, (void *)&pos567_1e8, products.ea_d, (void *)&temperature_kelvin_pow_4, products.only1_d, products.large_wave_radiation_atmosphere_d, tensors.stream));
    cudaEventRecord(stop);

    float cuda_time = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuda_time, start, stop);
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    return "CUTENSOR,LARGE_WAVE_RADIATION_ATMOSPHERE," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string net_radiation_function(Products products, Tensor tensors)
{
    int64_t initial_time, final_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    cudaEventRecord(start);
    float pos0 = 0;
    float pos1 = 1;
    float neg1 = -1;
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseTrinaryExecute(tensors.handle, tensors.tensor_plan_trinity_mult_add, (void *)&neg1, products.short_wave_radiation_d, (void *)&pos1, products.albedo_d, (void *)&pos1, products.short_wave_radiation_d, products.tensor_aux1_d, tensors.stream));
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add, (void *)&pos1, products.tensor_aux1_d, (void *)&pos1, products.large_wave_radiation_atmosphere_d, products.tensor_aux1_d, tensors.stream));
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add, (void *)&pos1, products.tensor_aux1_d, (void *)&neg1, products.large_wave_radiation_surface_d, products.tensor_aux1_d, tensors.stream));
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseTrinaryExecute(tensors.handle, tensors.tensor_plan_trinity_add_mult, (void *)&pos1, products.only1_d, (void *)&neg1, products.eo_d, (void *)&pos1, products.large_wave_radiation_atmosphere_d, products.tensor_aux2_d, tensors.stream));
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add, (void *)&pos1, products.tensor_aux1_d, (void *)&neg1, products.tensor_aux2_d, products.net_radiation_d, tensors.stream));
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_max, (void *)&pos1, products.net_radiation_d, (void *)&pos0, products.only1_d, products.net_radiation_d, tensors.stream));
    cudaEventRecord(stop);

    float cuda_time = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuda_time, start, stop);
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    return "CUTENSOR,NET_RADIATION," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string soil_heat_flux_function(Products products, Tensor tensors)
{
    int64_t initial_time, final_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    cudaEventRecord(start);
    float pos0 = 0;
    float pos1 = 1;
    float pos0074 = 0.0074;
    float post0038 = 0.0038;
    float neg27315 = -273.15;
    float neg098 = -0.98;
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_mult, (void *)&pos1, products.ndvi_d, (void *)&pos1, products.ndvi_d, products.tensor_aux2_d, tensors.stream));
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_mult, (void *)&pos1, products.tensor_aux2_d, (void *)&pos1, products.tensor_aux2_d, products.tensor_aux2_d, tensors.stream));
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add, (void *)&pos1, products.only1_d, (void *)&neg098, products.tensor_aux2_d, products.tensor_aux2_d, tensors.stream));
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add, (void *)&post0038, products.only1_d, (void *)&pos0074, products.albedo_d, products.tensor_aux1_d, tensors.stream));
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseTrinaryExecute(tensors.handle, tensors.tensor_plan_trinity_add_mult, (void *)&pos1, products.surface_temperature_d, (void *)&neg27315, products.only1_d, (void *)&pos1, products.tensor_aux1_d, products.tensor_aux1_d, tensors.stream));
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_mult, (void *)&pos1, products.tensor_aux1_d, (void *)&pos1, products.tensor_aux2_d, products.tensor_aux1_d, tensors.stream));
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_mult, (void *)&pos1, products.tensor_aux1_d, (void *)&pos1, products.net_radiation_d, products.soil_heat_d, tensors.stream));
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_max, (void *)&pos1, products.soil_heat_d, (void *)&pos0, products.only1_d, products.soil_heat_d, tensors.stream));
    cudaEventRecord(stop);

    float cuda_time = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuda_time, start, stop);
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    return "CUTENSOR,SOIL_HEAT_FLUX," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::compute_Rn_G(Products products, Station station, MTL mtl, Tensor tensors)
{
    string result = "";
    int64_t initial_time, final_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    cudaEventRecord(start);
    result += radiance_function(products, mtl, tensors);
    result += reflectance_function(products, mtl, tensors);
    result += albedo_function(products, mtl, tensors);

    // Vegetation indices
    result += ndvi_function(products, tensors);
    result += pai_function(products, tensors);
    result += lai_function(products, tensors);

    // Emissivity indices
    result += enb_emissivity_function(products, tensors);
    result += eo_emissivity_function(products, tensors);
    result += ea_emissivity_function(products, tensors);
    result += surface_temperature_function(products, mtl, tensors);

    // Radiation waves
    result += short_wave_radiation_function(products, mtl, tensors);
    result += large_wave_radiation_surface_function(products, tensors);
    result += large_wave_radiation_atmosphere_function(products, tensors, station.temperature_image);

    // Main products
    result += net_radiation_function(products, tensors);
    result += soil_heat_flux_function(products, tensors);
    cudaEventRecord(stop);

    float cuda_time = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuda_time, start, stop);
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    result += "KERNELS,P1_INITIAL_PROD," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
    return result;
}
