#include "constants.h"
#include "cuda_utils.h"
#include "kernels.cuh"
#include "sensors.cuh"
#include "surfaceData.cuh"

string rad_ref_combined_function(Products products, MTL mtl)
{
    const float sin_sun = sin(mtl.sun_elevation * PI / 180);

    int64_t initial_time, final_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    cudaEventRecord(start);

    // Executar kernels combinados em streams paralelos
    cudaStreamWaitEvent(products.stream_1, products.copy_done_events[0], 0);
    ref_kernel<<<blocks_n, threads_n, 0, products.stream_1>>>(products.band_blue_d, products.reflectance_blue_d, mtl.ref_add_d, mtl.ref_mult_d, sin_sun, PARAM_BAND_BLUE_INDEX);
    
    cudaStreamWaitEvent(products.stream_2, products.copy_done_events[1], 0);
    ref_kernel<<<blocks_n, threads_n, 0, products.stream_2>>>(products.band_green_d, products.reflectance_green_d, mtl.ref_add_d, mtl.ref_mult_d, sin_sun, PARAM_BAND_GREEN_INDEX);
    
    cudaStreamWaitEvent(products.stream_3, products.copy_done_events[2], 0);
    ref_kernel<<<blocks_n, threads_n, 0, products.stream_3>>>(products.band_red_d, products.reflectance_red_d, mtl.ref_add_d, mtl.ref_mult_d, sin_sun, PARAM_BAND_RED_INDEX);
    
    cudaStreamWaitEvent(products.stream_4, products.copy_done_events[3], 0);
    ref_kernel<<<blocks_n, threads_n, 0, products.stream_4>>>(products.band_nir_d, products.reflectance_nir_d, mtl.ref_add_d, mtl.ref_mult_d, sin_sun, PARAM_BAND_NIR_INDEX);
    
    cudaStreamWaitEvent(products.stream_5, products.copy_done_events[4], 0);
    ref_kernel<<<blocks_n, threads_n, 0, products.stream_5>>>(products.band_swir1_d, products.reflectance_swir1_d, mtl.ref_add_d, mtl.ref_mult_d, sin_sun, PARAM_BAND_SWIR1_INDEX);
    
    cudaStreamWaitEvent(products.stream_6, products.copy_done_events[5], 0);
    rad_kernel<<<blocks_n, threads_n, 0, products.stream_6>>>(products.band_termal_d, products.radiance_termal_d, mtl.rad_add_d, mtl.rad_mult_d, PARAM_BAND_TERMAL_INDEX);
    ref_kernel<<<blocks_n, threads_n, 0, products.stream_6>>>(products.band_termal_d, products.reflectance_termal_d, mtl.ref_add_d, mtl.ref_mult_d, sin_sun, PARAM_BAND_TERMAL_INDEX);

    cudaStreamWaitEvent(products.stream_7, products.copy_done_events[6], 0);
    ref_kernel<<<blocks_n, threads_n, 0, products.stream_7>>>(products.band_swir2_d, products.reflectance_swir2_d, mtl.ref_add_d, mtl.ref_mult_d, sin_sun, PARAM_BAND_SWIR2_INDEX);

    cudaEventRecord(stop);

    HANDLE_ERROR(cudaStreamSynchronize(products.stream_1));
    HANDLE_ERROR(cudaStreamSynchronize(products.stream_2));
    HANDLE_ERROR(cudaStreamSynchronize(products.stream_3));
    HANDLE_ERROR(cudaStreamSynchronize(products.stream_4));
    HANDLE_ERROR(cudaStreamSynchronize(products.stream_5));
    HANDLE_ERROR(cudaStreamSynchronize(products.stream_6));
    HANDLE_ERROR(cudaStreamSynchronize(products.stream_7));
    HANDLE_ERROR(cudaStreamSynchronize(products.stream_8));

    float cuda_time = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuda_time, start, stop);
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    return "KERNELS,RAD_REF," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
}

string alb_nvdi_lai_pai_emissivity_function(Products products, MTL mtl)
{
    int64_t initial_time, final_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    cudaEventRecord(start);

    // ========== Compute this products asynchronously with streams (6, 7, 8)
    ndvi_kernel<<<blocks_n, threads_n, 0, products.stream_6>>>(products.reflectance_nir_d, products.reflectance_red_d, products.ndvi_d);
    lai_kernel<<<blocks_n, threads_n, 0, products.stream_6>>>(products.reflectance_nir_d, products.reflectance_red_d, products.lai_d);
    if (model_method == 0)
    {
        pai_kernel<<<blocks_n, threads_n, 0, products.stream_6>>>(products.reflectance_nir_d, products.reflectance_red_d, products.pai_d);
    }

    albedo_kernel<<<blocks_n, threads_n, 0, products.stream_7>>>(
        products.reflectance_blue_d, products.reflectance_green_d, products.reflectance_red_d,
        products.reflectance_nir_d, products.reflectance_swir1_d, products.reflectance_swir2_d,
        products.tal_d, products.albedo_d, mtl.ref_w_coeff_d);

    enb_kernel<<<blocks_n, threads_n, 0, products.stream_8>>>(products.lai_d, products.ndvi_d, products.enb_d);
    eo_kernel<<<blocks_n, threads_n, 0, products.stream_8>>>(products.lai_d, products.ndvi_d, products.eo_d);
    ea_kernel<<<blocks_n, threads_n, 0, products.stream_8>>>(products.tal_d, products.ea_d);

    cudaEventRecord(stop);

    HANDLE_ERROR(cudaStreamSynchronize(products.stream_6));
    HANDLE_ERROR(cudaStreamSynchronize(products.stream_7));
    HANDLE_ERROR(cudaStreamSynchronize(products.stream_8));

    float cuda_time = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuda_time, start, stop);
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    return "KERNELS,ALB_NDVI_LAI_PAI_E," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
}

string surface_shotwave_function(Products products, MTL mtl)
{
    int64_t initial_time, final_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    cudaEventRecord(start);

    // ========== Compute this products asynchronously with streams (6, 7)
    surface_temperature_kernel<<<blocks_n, threads_n, 0, products.stream_6>>>(
        products.enb_d, products.radiance_termal_d, products.surface_temperature_d,
        (mtl.number_sensor == 5) ? 607.76f : (mtl.number_sensor == 7) ? 666.09f
                                                                      : 774.8853f,
        (mtl.number_sensor == 5) ? 1260.56f : (mtl.number_sensor == 7) ? 1282.71f
                                                                       : 1321.0789f);

    short_wave_radiation_kernel<<<blocks_n, threads_n, 0, products.stream_7>>>(
        products.tal_d, products.short_wave_radiation_d, mtl.sun_elevation,
        mtl.distance_earth_sun, PI);

    cudaEventRecord(stop);

    HANDLE_ERROR(cudaStreamSynchronize(products.stream_6));
    HANDLE_ERROR(cudaStreamSynchronize(products.stream_7));

    float cuda_time = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuda_time, start, stop);
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    return "KERNELS,SURFACE_TEMP_SHORT_WAVE," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
}

string largewave_function(Products products, MTL mtl, Station station)
{
    int64_t initial_time, final_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    cudaEventRecord(start);

    large_waves_radiation_kernel<<<blocks_n, threads_n>>>(
        products.surface_temperature_d, products.eo_d, products.ea_d,
        products.large_wave_radiation_surface_d, products.large_wave_radiation_atmosphere_d,
        station.temperature_image);

    cudaEventRecord(stop);

    HANDLE_ERROR(cudaStreamSynchronize(products.stream_6));
    HANDLE_ERROR(cudaStreamSynchronize(products.stream_7));

    float cuda_time = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuda_time, start, stop);
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    return "KERNELS,LARGE_WAVES_RADIATION," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
}

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
    cudaEventRecord(start);

    result += rad_ref_combined_function(products, mtl);

    // Vegetation indices & Emissivity indices
    result += alb_nvdi_lai_pai_emissivity_function(products, mtl);

    // Radiation waves
    result += surface_shotwave_function(products, mtl);
    result += largewave_function(products, mtl, station);

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