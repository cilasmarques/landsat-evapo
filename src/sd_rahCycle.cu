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
    unsigned int hot_pos = hotCandidate.line * products.width_band + hotCandidate.col;
    unsigned int cold_pos = coldCandidate.line * products.width_band + coldCandidate.col;

    float ndvi_hot, ndvi_cold;
    HANDLE_ERROR(cudaMemcpy(&ndvi_hot, products.ndvi_d + hot_pos, sizeof(float), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(&ndvi_cold, products.ndvi_d + cold_pos, sizeof(float), cudaMemcpyDeviceToHost));

    float temperature_hot, temperature_cold;
    HANDLE_ERROR(cudaMemcpy(&temperature_hot, products.surface_temperature_d + hot_pos, sizeof(float), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(&temperature_cold, products.surface_temperature_d + cold_pos, sizeof(float), cudaMemcpyDeviceToHost));

    float net_radiation_hot, net_radiation_cold;
    HANDLE_ERROR(cudaMemcpy(&net_radiation_hot, products.net_radiation_d + hot_pos, sizeof(float), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(&net_radiation_cold, products.net_radiation_d + cold_pos, sizeof(float), cudaMemcpyDeviceToHost));
    
    float soil_heat_flux_hot, soil_heat_flux_cold;
    HANDLE_ERROR(cudaMemcpy(&soil_heat_flux_hot, products.soil_heat_d + hot_pos, sizeof(float), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(&soil_heat_flux_cold, products.soil_heat_d + cold_pos, sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < 2; i++) {
        float rah_ini_hot, rah_ini_cold;
        HANDLE_ERROR(cudaMemcpy(&rah_ini_hot, products.rah_d + hot_pos, sizeof(float), cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(&rah_ini_cold, products.rah_d + cold_pos, sizeof(float), cudaMemcpyDeviceToHost));

        float fc_hot = 1.0f - powf((ndvi_hot - ndvi_max) / (ndvi_min - ndvi_max), 0.4631f);
        float fc_cold = 1.0f - powf((ndvi_cold - ndvi_max) / (ndvi_min - ndvi_max), 0.4631f);

        float LE_hot = 0.55f * fc_hot * (net_radiation_hot - soil_heat_flux_hot) * 0.78f;
        float LE_cold = 1.75f * fc_cold * (net_radiation_cold - soil_heat_flux_cold) * 0.78f;

        float H_cold = net_radiation_cold - soil_heat_flux_cold - LE_hot;
        float dt_cold = H_cold * rah_ini_cold / (RHO * SPECIFIC_HEAT_AIR);

        float H_hot = net_radiation_hot - soil_heat_flux_hot - LE_cold;
        float dt_hot = H_hot * rah_ini_hot / (RHO * SPECIFIC_HEAT_AIR);

        float b = (dt_hot - dt_cold) / (temperature_hot - temperature_cold);
        float a = dt_cold - (b * temperature_cold);

        HANDLE_ERROR(cudaMemcpyToSymbol(a_d, &a, sizeof(float), 0, cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpyToSymbol(b_d, &b, sizeof(float), 0, cudaMemcpyHostToDevice));

        rah_correction_cycle_STEEP<<<blocks_n, threads_n>>>(products.surface_temperature_d, products.d0_d, products.kb1_d, products.zom_d, products.ustar_d, products.rah_d, products.sensible_heat_flux_d);
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
    unsigned int hot_pos = hotCandidate.line * products.width_band + hotCandidate.col;
    unsigned int cold_pos = coldCandidate.line * products.width_band + coldCandidate.col;

    float temperature_hot, temperature_cold;
    HANDLE_ERROR(cudaMemcpy(&temperature_hot, products.surface_temperature_d + hot_pos, sizeof(float), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(&temperature_cold, products.surface_temperature_d + cold_pos, sizeof(float), cudaMemcpyDeviceToHost));

    float net_radiation_hot, net_radiation_cold;
    HANDLE_ERROR(cudaMemcpy(&net_radiation_hot, products.net_radiation_d + hot_pos, sizeof(float), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(&net_radiation_cold, products.net_radiation_d + cold_pos, sizeof(float), cudaMemcpyDeviceToHost));
    
    float soil_heat_flux_hot, soil_heat_flux_cold;
    HANDLE_ERROR(cudaMemcpy(&soil_heat_flux_hot, products.soil_heat_d + hot_pos, sizeof(float), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(&soil_heat_flux_cold, products.soil_heat_d + cold_pos, sizeof(float), cudaMemcpyDeviceToHost));

    int i = 0;
    while (true) {
        float rah_ini_hot, rah_ini_cold, rah_hot_new;
        HANDLE_ERROR(cudaMemcpy(&rah_ini_hot, products.rah_d + hot_pos, sizeof(float), cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(&rah_ini_cold, products.rah_d + cold_pos, sizeof(float), cudaMemcpyDeviceToHost));
        
        float H_cold = net_radiation_cold - soil_heat_flux_cold;
        float dt_cold = H_cold * rah_ini_cold / (RHO * SPECIFIC_HEAT_AIR);

        float H_hot = net_radiation_hot - soil_heat_flux_hot;
        float dt_hot = H_hot * rah_ini_hot / (RHO * SPECIFIC_HEAT_AIR);

        float b = (dt_hot - dt_cold) / (temperature_hot - temperature_cold);
        float a = dt_cold - (b * temperature_cold);

        HANDLE_ERROR(cudaMemcpyToSymbol(a_d, &a, sizeof(float), 0, cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpyToSymbol(b_d, &b, sizeof(float), 0, cudaMemcpyHostToDevice));

        rah_correction_cycle_ASEBAL<<<blocks_n, threads_n>>>(products.surface_temperature_d, products.zom_d, products.ustar_d, products.rah_d, products.sensible_heat_flux_d, u200);

        HANDLE_ERROR(cudaMemcpy(&rah_hot_new, products.rah_d + hot_pos, sizeof(float), cudaMemcpyDeviceToHost));

        if ((i > 0) && (fabsf(1.0f - (rah_ini_hot / rah_hot_new)) < 0.05f)) 
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
