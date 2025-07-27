#include "cuda_utils.h"
#include "kernels.cuh"
#include "sensors.cuh"
#include "surfaceData.cuh"

string d0_fuction(Products products)
{
    float CD1 = 20.6;
    float HGHT = 4;

    int64_t initial_time, final_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    cudaEventRecord(start);
    d0_kernel<<<blocks_n, threads_n>>>(products.pai_d, products.d0_d, CD1, HGHT);
    cudaEventRecord(stop);

    float cuda_time = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuda_time, start, stop);
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    return "KERNELS,D0," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string kb_function(Products products, float ndvi_max, float ndvi_min)
{
    int64_t initial_time, final_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    cudaEventRecord(start);
    kb_kernel<<<blocks_n, threads_n>>>(products.zom_d, products.ustar_d, products.pai_d, products.kb1_d, products.ndvi_d, ndvi_max, ndvi_min);
    cudaEventRecord(stop);

    float cuda_time = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuda_time, start, stop);
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    return "KERNELS,KB1," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string zom_fuction(Products products, float A_ZOM, float B_ZOM)
{
    int64_t initial_time, final_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    cudaEventRecord(start);
    if (model_method == 0)
        zom_kernel_STEEP<<<blocks_n, threads_n>>>(products.d0_d, products.pai_d, products.zom_d, A_ZOM, B_ZOM);
    else
        zom_kernel_ASEBAL<<<blocks_n, threads_n>>>(products.ndvi_d, products.albedo_d, products.zom_d, A_ZOM, B_ZOM);
    cudaEventRecord(stop);

    float cuda_time = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuda_time, start, stop);
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    return "KERNELS,ZOM," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string ustar_fuction(Products products, float u_const)
{
    int64_t initial_time, final_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    cudaEventRecord(start);
    if (model_method == 0)
        ustar_kernel_STEEP<<<blocks_n, threads_n>>>(products.zom_d, products.d0_d, products.ustar_d, u_const);
    else
        ustar_kernel_ASEBAL<<<blocks_n, threads_n>>>(products.zom_d, products.ustar_d, u_const);
    cudaEventRecord(stop);

    float cuda_time = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuda_time, start, stop);
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    return "KERNELS,USTAR," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string aerodynamic_resistance_fuction(Products products)
{
    int64_t initial_time, final_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    cudaEventRecord(start);
    if (model_method == 0)
        aerodynamic_resistance_kernel_STEEP<<<blocks_n, threads_n>>>(products.zom_d, products.d0_d, products.ustar_d, products.kb1_d, products.rah_d);
    else
        aerodynamic_resistance_kernel_ASEBAL<<<blocks_n, threads_n>>>(products.ustar_d, products.rah_d);
    cudaEventRecord(stop);

    float cuda_time = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuda_time, start, stop);
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    return "KERNELS,RAH_INI," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string rah_correction_function_blocks_STEEP(Products products, float ndvi_min, float ndvi_max)
{
    string result = "";
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    int64_t initial_time, final_time;

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    // ========= CUDA Setup
    int dev = 0;
    cudaDeviceProp deviceProp;
    HANDLE_ERROR(cudaGetDeviceProperties(&deviceProp, dev));
    HANDLE_ERROR(cudaSetDevice(dev));

    cudaEventRecord(start);
    for (int i = 0; i < 2; i++) {
        rah_correction_cycle_STEEP<<<blocks_n, threads_n>>>(products.net_radiation_d, products.soil_heat_d, products.ndvi_d, products.surface_temperature_d, products.d0_d, products.kb1_d, products.zom_d, products.ustar_d, products.rah_d, products.sensible_heat_flux_d, ndvi_max, ndvi_min);
    }
    cudaEventRecord(stop);

    float cuda_time = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuda_time, start, stop);
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
    return "KERNELS,RAH_CYCLE," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
}

string rah_correction_function_blocks_ASEBAL(Products products, float u200)
{
    string result = "";
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    int64_t initial_time, final_time;

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    // ========= CUDA Setup
    int dev = 0;
    cudaDeviceProp deviceProp;
    HANDLE_ERROR(cudaGetDeviceProperties(&deviceProp, dev));
    HANDLE_ERROR(cudaSetDevice(dev));

    cudaEventRecord(start);
    int i = 0;
    while (true) {
        rah_correction_cycle_ASEBAL<<<blocks_n, threads_n>>>(products.net_radiation_d, products.soil_heat_d, products.ndvi_d, products.surface_temperature_d, products.kb1_d, products.zom_d, products.ustar_d, products.rah_d, products.sensible_heat_flux_d, u200, products.stop_condition_d);

        HANDLE_ERROR(cudaMemcpy(products.stop_condition, products.stop_condition_d, sizeof(int), cudaMemcpyDeviceToHost));

        if (i > 0 && *products.stop_condition)
            break;
        else
            i++;
    }
    cudaEventRecord(stop);

    float cuda_time = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuda_time, start, stop);
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
    return "KERNELS,RAH_CYCLE," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
}


string sensible_heat_flux_function(Products products)
{
    int64_t initial_time, final_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    cudaEventRecord(start);
    sensible_heat_flux_kernel<<<blocks_n, threads_n>>>(products.surface_temperature_d, products.rah_d, products.net_radiation_d, products.soil_heat_d, products.sensible_heat_flux_d);
    cudaEventRecord(stop);

    float cuda_time = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuda_time, start, stop);
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    return "KERNELS,SENSIBLE_HEAT_FLUX," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::converge_rah_cycle(Products products, Station station)
{
    string result = "";
    int64_t initial_time, final_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    cudaEventRecord(start);
    float ustar_station = (VON_KARMAN * station.v6) / (logf(station.WIND_SPEED / station.SURFACE_ROUGHNESS));
    float u10 = (ustar_station / VON_KARMAN) * logf(10.0f / station.SURFACE_ROUGHNESS);
    float u200 = (ustar_station / VON_KARMAN) * logf(200.0f / station.SURFACE_ROUGHNESS);

    thrust::device_ptr<float> ndvi_ptr = thrust::device_pointer_cast(products.ndvi_d);
    
    float ndvi_min = thrust::reduce(ndvi_ptr, 
                                   ndvi_ptr + products.height_band * products.width_band,
                                   1.0f, // Initial value
                                   min_valid());
    
    float ndvi_max = thrust::reduce(ndvi_ptr, 
                                   ndvi_ptr + products.height_band * products.width_band,
                                   -1.0f, // Initial value
                                   max_valid());

    if (model_method == 0) { // STEEP
        result += d0_fuction(products);
        result += zom_fuction(products, station.A_ZOM, station.B_ZOM);
        result += ustar_fuction(products, u10);
        result += kb_function(products, ndvi_max, ndvi_min);
        result += aerodynamic_resistance_fuction(products);
        result += rah_correction_function_blocks_STEEP(products, ndvi_min, ndvi_max);
        result += sensible_heat_flux_function(products);
    } else { // ASEBAL
        result += zom_fuction(products, station.A_ZOM, station.B_ZOM);
        result += ustar_fuction(products, u200);
        result += aerodynamic_resistance_fuction(products);
        result += rah_correction_function_blocks_ASEBAL(products, u200);
        result += sensible_heat_flux_function(products);
    }
    cudaEventRecord(stop);

    float cuda_time = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuda_time, start, stop);
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    result += "KERNELS,P3_RAH," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
    return result;
};
