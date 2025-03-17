#include "constants.h"
#include "cuda_utils.h"
#include "kernels.cuh"
#include "sensors.cuh"
#include "surfaceData.cuh"

string radiance_function(Products products, MTL mtl)
{
    int64_t initial_time, final_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    cudaEventRecord(start, 0);
    // rad_kernel<<<blocks_n, threads_n>>>(products.band_blue_d, products.radiance_blue_d, mtl.rad_add_d, mtl.rad_mult_d, PARAM_BAND_BLUE_INDEX);
    // rad_kernel<<<blocks_n, threads_n>>>(products.band_green_d, products.radiance_green_d, mtl.rad_add_d, mtl.rad_mult_d, PARAM_BAND_GREEN_INDEX);
    // rad_kernel<<<blocks_n, threads_n>>>(products.band_red_d, products.radiance_red_d, mtl.rad_add_d, mtl.rad_mult_d, PARAM_BAND_RED_INDEX);
    // rad_kernel<<<blocks_n, threads_n>>>(products.band_nir_d, products.radiance_nir_d, mtl.rad_add_d, mtl.rad_mult_d, PARAM_BAND_NIR_INDEX);
    // rad_kernel<<<blocks_n, threads_n>>>(products.band_swir1_d, products.radiance_swir1_d, mtl.rad_add_d, mtl.rad_mult_d, PARAM_BAND_SWIR1_INDEX);
    // rad_kernel<<<blocks_n, threads_n>>>(products.band_swir2_d, products.radiance_swir2_d, mtl.rad_add_d, mtl.rad_mult_d, PARAM_BAND_SWIR2_INDEX);
    rad_kernel<<<blocks_n, threads_n>>>(products.band_termal_d, products.radiance_termal_d, mtl.rad_add_d, mtl.rad_mult_d, PARAM_BAND_TERMAL_INDEX);
    cudaEventRecord(stop, 0);

    float cuda_time = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuda_time, start, stop);
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    return "KERNELS,RADIANCE," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
}

string reflectance_function(Products products, MTL mtl)
{
    const float sin_sun = sin(mtl.sun_elevation * PI / 180);

    int64_t initial_time, final_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    cudaEventRecord(start);
    ref_kernel<<<blocks_n, threads_n>>>(products.band_blue_d, products.reflectance_blue_d, mtl.ref_add_d, mtl.ref_mult_d, sin_sun, PARAM_BAND_BLUE_INDEX);
    ref_kernel<<<blocks_n, threads_n>>>(products.band_green_d, products.reflectance_green_d, mtl.ref_add_d, mtl.ref_mult_d, sin_sun, PARAM_BAND_GREEN_INDEX);
    ref_kernel<<<blocks_n, threads_n>>>(products.band_red_d, products.reflectance_red_d, mtl.ref_add_d, mtl.ref_mult_d, sin_sun, PARAM_BAND_RED_INDEX);
    ref_kernel<<<blocks_n, threads_n>>>(products.band_nir_d, products.reflectance_nir_d, mtl.ref_add_d, mtl.ref_mult_d, sin_sun, PARAM_BAND_NIR_INDEX);
    ref_kernel<<<blocks_n, threads_n>>>(products.band_swir1_d, products.reflectance_swir1_d, mtl.ref_add_d, mtl.ref_mult_d, sin_sun, PARAM_BAND_SWIR1_INDEX);
    ref_kernel<<<blocks_n, threads_n>>>(products.band_termal_d, products.reflectance_termal_d, mtl.ref_add_d, mtl.ref_mult_d, sin_sun, PARAM_BAND_TERMAL_INDEX);
    ref_kernel<<<blocks_n, threads_n>>>(products.band_swir2_d, products.reflectance_swir2_d, mtl.ref_add_d, mtl.ref_mult_d, sin_sun, PARAM_BAND_SWIR2_INDEX);
    cudaEventRecord(stop);

    float cuda_time = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuda_time, start, stop);
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    return "KERNELS,REFLECTANCE," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
}

string albedo_function(Products products, MTL mtl)
{
    int64_t initial_time, final_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    cudaEventRecord(start);
    albedo_kernel<<<blocks_n, threads_n>>>(products.reflectance_blue_d, products.reflectance_green_d, products.reflectance_red_d, products.reflectance_nir_d, products.reflectance_swir1_d, products.reflectance_swir2_d, products.tal_d, products.albedo_d, mtl.ref_w_coeff_d);
    cudaEventRecord(stop);

    float cuda_time = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuda_time, start, stop);
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    return "KERNELS,ALBEDO," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
}

string ndvi_function(Products products)
{
    int64_t initial_time, final_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    cudaEventRecord(start);
    ndvi_kernel<<<blocks_n, threads_n>>>(products.reflectance_nir_d, products.reflectance_red_d, products.ndvi_d);
    cudaEventRecord(stop);

    float cuda_time = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuda_time, start, stop);
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    return "KERNELS,NDVI," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string pai_function(Products products)
{
    int64_t initial_time, final_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    cudaEventRecord(start);
    pai_kernel<<<blocks_n, threads_n>>>(products.reflectance_nir_d, products.reflectance_red_d, products.pai_d);
    cudaEventRecord(stop);

    float cuda_time = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuda_time, start, stop);
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    return "KERNELS,PAI," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string lai_function(Products products)
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

string enb_emissivity_function(Products products)
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

string eo_emissivity_function(Products products)
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

string ea_emissivity_function(Products products)
{
    int64_t initial_time, final_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    cudaEventRecord(start);
    ea_kernel<<<blocks_n, threads_n>>>(products.tal_d, products.ea_d);
    cudaEventRecord(stop);

    float cuda_time = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuda_time, start, stop);
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    return "KERNELS,EA_EMISSIVITY," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string surface_temperature_function(Products products, MTL mtl)
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
    surface_temperature_kernel<<<blocks_n, threads_n>>>(products.enb_d, products.radiance_termal_d, products.surface_temperature_d, k1, k2);
    cudaEventRecord(stop);

    float cuda_time = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuda_time, start, stop);
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    return "KERNELS,SURFACE_TEMPERATURE," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string short_wave_radiation_function(Products products, MTL mtl)
{
    int64_t initial_time, final_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    cudaEventRecord(start);
    short_wave_radiation_kernel<<<blocks_n, threads_n>>>(products.tal_d, products.short_wave_radiation_d, mtl.sun_elevation, mtl.distance_earth_sun, PI);
    cudaEventRecord(stop);

    float cuda_time = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuda_time, start, stop);
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    return "KERNELS,SHORT_WAVE_RADIATION," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string large_wave_radiation_surface_function(Products products)
{
    int64_t initial_time, final_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    cudaEventRecord(start);
    large_wave_radiation_surface_kernel<<<blocks_n, threads_n>>>(products.surface_temperature_d, products.eo_d, products.large_wave_radiation_surface_d);
    cudaEventRecord(stop);

    float cuda_time = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuda_time, start, stop);
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    return "KERNELS,LARGE_WAVE_RADIATION_SURFACE," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string large_wave_radiation_atmosphere_function(Products products, float temperature)
{
    int64_t initial_time, final_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    cudaEventRecord(start);
    large_wave_radiation_atmosphere_kernel<<<blocks_n, threads_n>>>(products.ea_d, products.large_wave_radiation_atmosphere_d, temperature);
    cudaEventRecord(stop);

    float cuda_time = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuda_time, start, stop);
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    return "KERNELS,LARGE_WAVE_RADIATION_ATMOSPHERE," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string net_radiation_function(Products products)
{
    int64_t initial_time, final_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    cudaEventRecord(start);
    net_radiation_kernel<<<blocks_n, threads_n>>>(products.short_wave_radiation_d, products.albedo_d, products.large_wave_radiation_atmosphere_d, products.large_wave_radiation_surface_d, products.eo_d, products.net_radiation_d);
    cudaEventRecord(stop);

    float cuda_time = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuda_time, start, stop);
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    return "KERNELS,NET_RADIATION," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string soil_heat_flux_function(Products products)
{
    int64_t initial_time, final_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    cudaEventRecord(start);
    soil_heat_kernel<<<blocks_n, threads_n>>>(products.ndvi_d, products.albedo_d, products.surface_temperature_d, products.net_radiation_d, products.soil_heat_d);
    cudaEventRecord(stop);

    float cuda_time = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuda_time, start, stop);
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    return "KERNELS,SOIL_HEAT_FLUX," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::compute_Rn_G(Products products, Station station, MTL mtl)
{
    string result = "";
    int64_t initial_time, final_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    HANDLE_ERROR(cudaStreamSynchronize(products.stream_blue));
    HANDLE_ERROR(cudaStreamSynchronize(products.stream_green));
    HANDLE_ERROR(cudaStreamSynchronize(products.stream_red));
    HANDLE_ERROR(cudaStreamSynchronize(products.stream_nir));
    HANDLE_ERROR(cudaStreamSynchronize(products.stream_swir1));
    HANDLE_ERROR(cudaStreamSynchronize(products.stream_termal));
    HANDLE_ERROR(cudaStreamSynchronize(products.stream_swir2));
    HANDLE_ERROR(cudaStreamSynchronize(products.stream_tal));

    HANDLE_ERROR(cudaStreamDestroy(products.stream_blue));
    HANDLE_ERROR(cudaStreamDestroy(products.stream_green));
    HANDLE_ERROR(cudaStreamDestroy(products.stream_red));
    HANDLE_ERROR(cudaStreamDestroy(products.stream_nir));
    HANDLE_ERROR(cudaStreamDestroy(products.stream_swir1));
    HANDLE_ERROR(cudaStreamDestroy(products.stream_termal));
    HANDLE_ERROR(cudaStreamDestroy(products.stream_swir2));
    HANDLE_ERROR(cudaStreamDestroy(products.stream_tal));


    cudaEventRecord(start);
    result += radiance_function(products, mtl);
    result += reflectance_function(products, mtl);
    result += albedo_function(products, mtl);

    // Vegetation indices
    result += ndvi_function(products);
    result += lai_function(products);
    if (model_method == 0)
        result += pai_function(products);

    // Emissivity indices
    result += enb_emissivity_function(products);
    result += eo_emissivity_function(products);
    result += ea_emissivity_function(products);
    result += surface_temperature_function(products, mtl);

    // Radiation waves
    result += short_wave_radiation_function(products, mtl);
    result += large_wave_radiation_surface_function(products);
    result += large_wave_radiation_atmosphere_function(products, station.temperature_image);

    // Main products
    result += net_radiation_function(products);
    result += soil_heat_flux_function(products);
    cudaEventRecord(stop);

    float cuda_time = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuda_time, start, stop);
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    result += "KERNELS,P1_INITIAL_PROD," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
    return result;
}
