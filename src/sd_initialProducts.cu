#include "constants.h"
#include "cuda_utils.h"
#include "kernels.cuh"
#include "sensors.cuh"
#include "surfaceData.cuh"
#include "utils.cuh"

string rad_ref_combined_function(Products products, MTL mtl)
{
    const float sin_sun = sin(mtl.sun_elevation * PI / 180);

    int64_t initial_time, final_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    cudaStream_t streams[7];
    for (int i = 0; i < 7; i++) {
        HANDLE_ERROR(cudaStreamCreate(&streams[i]));
    }

    cudaEventRecord(start);
    
    // Executar kernels combinados em streams paralelos
    rad_ref_kernel<<<blocks_n, threads_n, 0, streams[0]>>>(products.band_blue_d, products.radiance_blue_d, products.reflectance_blue_d, 
                                     mtl.rad_add_d, mtl.rad_mult_d, mtl.ref_add_d, mtl.ref_mult_d, 
                                     sin_sun, PARAM_BAND_BLUE_INDEX);
    
    rad_ref_kernel<<<blocks_n, threads_n, 0, streams[1]>>>(products.band_green_d, products.radiance_green_d, products.reflectance_green_d, 
                                     mtl.rad_add_d, mtl.rad_mult_d, mtl.ref_add_d, mtl.ref_mult_d, 
                                     sin_sun, PARAM_BAND_GREEN_INDEX);
    
    rad_ref_kernel<<<blocks_n, threads_n, 0, streams[2]>>>(products.band_red_d, products.radiance_red_d, products.reflectance_red_d, 
                                     mtl.rad_add_d, mtl.rad_mult_d, mtl.ref_add_d, mtl.ref_mult_d, 
                                     sin_sun, PARAM_BAND_RED_INDEX);
    
    rad_ref_kernel<<<blocks_n, threads_n, 0, streams[3]>>>(products.band_nir_d, products.radiance_nir_d, products.reflectance_nir_d, 
                                     mtl.rad_add_d, mtl.rad_mult_d, mtl.ref_add_d, mtl.ref_mult_d, 
                                     sin_sun, PARAM_BAND_NIR_INDEX);
    
    rad_ref_kernel<<<blocks_n, threads_n, 0, streams[4]>>>(products.band_swir1_d, products.radiance_swir1_d, products.reflectance_swir1_d, 
                                     mtl.rad_add_d, mtl.rad_mult_d, mtl.ref_add_d, mtl.ref_mult_d, 
                                     sin_sun, PARAM_BAND_SWIR1_INDEX);
    
    rad_ref_kernel<<<blocks_n, threads_n, 0, streams[5]>>>(products.band_termal_d, products.radiance_termal_d, products.reflectance_termal_d, 
                                     mtl.rad_add_d, mtl.rad_mult_d, mtl.ref_add_d, mtl.ref_mult_d, 
                                     sin_sun, PARAM_BAND_TERMAL_INDEX);
    
    rad_ref_kernel<<<blocks_n, threads_n, 0, streams[6]>>>(products.band_swir2_d, products.radiance_swir2_d, products.reflectance_swir2_d, 
                                     mtl.rad_add_d, mtl.rad_mult_d, mtl.ref_add_d, mtl.ref_mult_d, 
                                     sin_sun, PARAM_BAND_SWIR2_INDEX);
    
    // Sincronizar todos os streams
    for (int i = 0; i < 7; i++) {
        HANDLE_ERROR(cudaStreamSynchronize(streams[i]));
        HANDLE_ERROR(cudaStreamDestroy(streams[i]));
    }
    
    cudaEventRecord(stop);

    float cuda_time = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuda_time, start, stop);
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    return "KERNELS,RAD_REF," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
}

string Products::compute_Rn_G(Products products, Station station, MTL mtl)
{
    string result = "";
    int64_t initial_time, final_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    HANDLE_ERROR(cudaStreamSynchronize(products.stream_1));
    HANDLE_ERROR(cudaStreamSynchronize(products.stream_2));
    HANDLE_ERROR(cudaStreamSynchronize(products.stream_3));
    HANDLE_ERROR(cudaStreamSynchronize(products.stream_4));
    HANDLE_ERROR(cudaStreamSynchronize(products.stream_5));
    HANDLE_ERROR(cudaStreamSynchronize(products.stream_6));
    HANDLE_ERROR(cudaStreamSynchronize(products.stream_7));
    HANDLE_ERROR(cudaStreamSynchronize(products.stream_8));

    HANDLE_ERROR(cudaStreamDestroy(products.stream_1));
    HANDLE_ERROR(cudaStreamDestroy(products.stream_2));
    HANDLE_ERROR(cudaStreamDestroy(products.stream_3));
    HANDLE_ERROR(cudaStreamDestroy(products.stream_4));
    HANDLE_ERROR(cudaStreamDestroy(products.stream_5));

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
    cudaEventRecord(start);
    
    result += rad_ref_combined_function(products, mtl);
    
    // ========== Compute this products asynchronously with streams (6, 7, 8)
    ndvi_kernel<<<blocks_n, threads_n, 0, products.stream_6>>>(products.reflectance_nir_d, products.reflectance_red_d, products.ndvi_d);
    lai_kernel<<<blocks_n, threads_n, 0, products.stream_6>>>(products.reflectance_nir_d, products.reflectance_red_d, products.lai_d);
    if (model_method == 0) {
        pai_kernel<<<blocks_n, threads_n, 0, products.stream_6>>>(products.reflectance_nir_d, products.reflectance_red_d, products.pai_d);
    }
    
    albedo_kernel<<<blocks_n, threads_n, 0, products.stream_7>>>(
        products.reflectance_blue_d, products.reflectance_green_d, products.reflectance_red_d, 
        products.reflectance_nir_d, products.reflectance_swir1_d, products.reflectance_swir2_d, 
        products.tal_d, products.albedo_d, mtl.ref_w_coeff_d
    );
    
    enb_kernel<<<blocks_n, threads_n, 0, products.stream_8>>>(products.lai_d, products.ndvi_d, products.enb_d);
    eo_kernel<<<blocks_n, threads_n, 0, products.stream_8>>>(products.lai_d, products.ndvi_d, products.eo_d);
    ea_kernel<<<blocks_n, threads_n, 0, products.stream_8>>>(products.tal_d, products.ea_d);

    HANDLE_ERROR(cudaStreamSynchronize(products.stream_6));
    HANDLE_ERROR(cudaStreamSynchronize(products.stream_7));
    HANDLE_ERROR(cudaStreamSynchronize(products.stream_8));

    // ========== Compute this products asynchronously with streams (6, 7) 
    surface_temperature_kernel<<<blocks_n, threads_n, 0, products.stream_6>>>(
        products.enb_d, products.radiance_termal_d, products.surface_temperature_d, 
        (mtl.number_sensor == 5) ? 607.76f : (mtl.number_sensor == 7) ? 666.09f : 774.8853f,
        (mtl.number_sensor == 5) ? 1260.56f : (mtl.number_sensor == 7) ? 1282.71f : 1321.0789f
    );

    short_wave_radiation_kernel<<<blocks_n, threads_n, 0, products.stream_7>>>(
        products.tal_d, products.short_wave_radiation_d, mtl.sun_elevation, 
        mtl.distance_earth_sun, PI
    );

    HANDLE_ERROR(cudaStreamSynchronize(products.stream_6));
    HANDLE_ERROR(cudaStreamSynchronize(products.stream_7));

    // ========== Compute this products synchronously 
    large_wave_radiation_combined_kernel<<<blocks_n, threads_n>>>(
        products.surface_temperature_d, products.eo_d, products.ea_d,
        products.large_wave_radiation_surface_d, products.large_wave_radiation_atmosphere_d,
        station.temperature_image
    );
    
    net_radiation_soil_heat_kernel<<<blocks_n, threads_n>>>(
        products.short_wave_radiation_d, products.albedo_d,
        products.large_wave_radiation_atmosphere_d, products.large_wave_radiation_surface_d,
        products.eo_d, products.ndvi_d, products.surface_temperature_d,
        products.net_radiation_d, products.soil_heat_d
    );

    HANDLE_ERROR(cudaStreamDestroy(products.stream_6));
    HANDLE_ERROR(cudaStreamDestroy(products.stream_7));
    HANDLE_ERROR(cudaStreamDestroy(products.stream_8));

    cudaEventRecord(stop);
    
    float cuda_time = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuda_time, start, stop);
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    result += "KERNELS,P1_INITIAL_PROD," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
    return result;
}
