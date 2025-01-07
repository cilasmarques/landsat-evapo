#include "cuda_utils.h"
#include "kernels.cuh"
#include "sensors.cuh"
#include "surfaceData.cuh"

string d0_fuction(Products products, Tensor tensors)
{
    int64_t initial_time, final_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    cudaEventRecord(start);
    float CD1 = 20.6;
    float HGHT = 4;
    float pos1 = 1;
    float neg1 = -1;
    HANDLE_CUTENSOR_ERROR(cutensorPermute(tensors.handle, tensors.tensor_plan_permute_id, (void *)&CD1, products.pai_d, products.tensor_aux1_d, tensors.stream));
    HANDLE_CUTENSOR_ERROR(cutensorPermute(tensors.handle, tensors.tensor_plan_permute_sqtr, (void *)&pos1, products.tensor_aux1_d, products.tensor_aux1_d, tensors.stream));
    HANDLE_CUTENSOR_ERROR(cutensorPermute(tensors.handle, tensors.tensor_plan_permute_exp, (void *)&pos1, products.only1_d, products.tensor_aux2_d, tensors.stream));
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_log_mul, (void *)&neg1, products.tensor_aux1_d, (void *)&pos1, products.tensor_aux2_d, products.tensor_aux2_d, tensors.stream));
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_div, (void *)&pos1, products.only1_d, (void *)&pos1, products.tensor_aux1_d, products.tensor_aux1_d, tensors.stream));
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_exp_mul, (void *)&pos1, products.tensor_aux1_d, (void *)&pos1, products.tensor_aux2_d, products.tensor_aux2_d, tensors.stream));
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add, (void *)&pos1, products.only1_d, (void *)&neg1, products.tensor_aux1_d, products.tensor_aux1_d, tensors.stream));
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add, (void *)&HGHT, products.tensor_aux1_d, (void *)&pos1, products.tensor_aux2_d, products.d0_d, tensors.stream));
    cudaEventRecord(stop);

    float cuda_time = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuda_time, start, stop);
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
    return "KERNELS,D0," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string kb_function(Products products, Tensor tensors, float ndvi_max, float ndvi_min)
{
    int64_t initial_time, final_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    cudaEventRecord(start);
    float HGHT = 4;

    float visc = 0.00001461;
    float pr = 0.71;
    float c1 = 0.320;
    float c2 = 0.264;
    float c3 = 15.1;
    float cd = 0.2;
    float ct = 0.01;
    float sf_c = 0.3;
    float sf_d = 2.5;
    float sf_e = 4.0;

    float pos1 = 1;
    float pos2 = 2;
    float pos025 = 0.25;
    float pos246 = 2.46;
    float pos009 = 0.009;
    float pos4631 = 0.4631;
    float neg05 = -0.5;
    float neg1 = -1;
    float neg2 = -2;

    float ct4 = 4 * ct;
    float cdc3 = cd * -c3;
    float divHGHT = 1 / HGHT;
    float div_visc = 1 / visc;
    float cdVON = cd * VON_KARMAN;
    float pow_pr = pow(pr, -0.667);
    float neg_ndvi_max = -ndvi_max;
    float soil_moisture_day_rel = 0.33;
    float div_ndvi_min_max = 1 / (ndvi_min - ndvi_max);
    float SF = sf_c + (1 / (1 + pow(exp(1.0), (sf_d - (sf_e * soil_moisture_day_rel)))));

    // float Re_star = (this->ustar[i] * 0.009) / visc;
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_mult, (void *)&pos009, products.ustar_d, (void *)&div_visc, products.only1_d, products.tensor_aux3_d, tensors.stream));

    // float Ct_star = pow(pr, -0.667) * pow(Re_star, -0.5); ~ pow_pr * exp(-0.5 * log(Re_star)); // 0.02%
    HANDLE_CUTENSOR_ERROR(cutensorPermute(tensors.handle, tensors.tensor_plan_permute_log, (void *)&neg05, products.tensor_aux3_d, products.tensor_aux4_d, tensors.stream));
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_exp_mul, (void *)&pow_pr, products.only1_d, (void *)&pos1, products.tensor_aux4_d, products.tensor_aux4_d, tensors.stream));

    // float beta = c1 - c2 * (exp(cdc3 * this->pai[i])); // OK
    HANDLE_CUTENSOR_ERROR(cutensorPermute(tensors.handle, tensors.tensor_plan_permute_id, (void *)&cdc3, products.pai_d, products.beta_d, tensors.stream));
    HANDLE_CUTENSOR_ERROR(cutensorPermute(tensors.handle, tensors.tensor_plan_permute_exp, (void *)&c2, products.beta_d, products.beta_d, tensors.stream));
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add, (void *)&c1, products.only1_d, (void *)&neg1, products.beta_d, products.beta_d, tensors.stream));

    // float nec_terra = (cd * this->pai[i]) / (beta * beta * 2);  // OK
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_mult, (void *)&pos2, products.beta_d, (void *)&pos1, products.beta_d, products.nec_terra_d, tensors.stream));
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_div, (void *)&cd, products.pai_d, (void *)&pos1, products.nec_terra_d, products.nec_terra_d, tensors.stream));

    // float kb1_fst_part = (cd * VON_KARMAN) / (4 * ct * beta * (1 - exp(nec_terra * -0.5)));
    HANDLE_CUTENSOR_ERROR(cutensorPermute(tensors.handle, tensors.tensor_plan_permute_id, (void *)&neg05, products.nec_terra_d, products.kb1_fst_part_d, tensors.stream));
    HANDLE_CUTENSOR_ERROR(cutensorPermute(tensors.handle, tensors.tensor_plan_permute_exp, (void *)&pos1, products.kb1_fst_part_d, products.kb1_fst_part_d, tensors.stream));
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add, (void *)&pos1, products.only1_d, (void *)&neg1, products.kb1_fst_part_d, products.kb1_fst_part_d, tensors.stream));
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_mult, (void *)&ct4, products.beta_d, (void *)&pos1, products.kb1_fst_part_d, products.kb1_fst_part_d, tensors.stream));
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_div, (void *)&cdVON, products.only1_d, (void *)&pos1, products.kb1_fst_part_d, products.kb1_fst_part_d, tensors.stream));

    // float kb1_sec_part = (beta * VON_KARMAN * (this->zom[i] / HGHT)) / Ct_star;
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_mult, (void *)&divHGHT, products.only1_d, (void *)&pos1, products.zom_d, products.kb1_sec_part_d, tensors.stream));
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_mult, (void *)&VON_KARMAN, products.beta_d, (void *)&pos1, products.kb1_sec_part_d, products.kb1_sec_part_d, tensors.stream));
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_div, (void *)&pos1, products.kb1_sec_part_d, (void *)&pos1, products.tensor_aux4_d, products.kb1_sec_part_d, tensors.stream));

    // float kb1s = (pow(Re_star, 0.25) * 2.46) - 2) ~ 2.46 * exp(0.25 * log(Re_star));
    HANDLE_CUTENSOR_ERROR(cutensorPermute(tensors.handle, tensors.tensor_plan_permute_log, (void *)&pos025, products.tensor_aux3_d, products.kb1s_d, tensors.stream));
    HANDLE_CUTENSOR_ERROR(cutensorPermute(tensors.handle, tensors.tensor_plan_permute_exp, (void *)&pos246, products.kb1s_d, products.kb1s_d, tensors.stream));
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add, (void *)&pos1, products.kb1s_d, (void *)&neg2, products.only1_d, products.kb1s_d, tensors.stream));

    // float fc = 1 - pow((this->ndvi[i] - ndvi_max) / (ndvi_min - ndvi_max), 0.4631);
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add, (void *)&pos1, products.ndvi_d, (void *)&neg_ndvi_max, products.only1_d, products.fc_d, tensors.stream));
    HANDLE_CUTENSOR_ERROR(cutensorPermute(tensors.handle, tensors.tensor_plan_permute_id, (void *)&div_ndvi_min_max, products.fc_d, products.fc_d, tensors.stream));
    HANDLE_CUTENSOR_ERROR(cutensorPermute(tensors.handle, tensors.tensor_plan_permute_log, (void *)&pos4631, products.fc_d, products.fc_d, tensors.stream));
    HANDLE_CUTENSOR_ERROR(cutensorPermute(tensors.handle, tensors.tensor_plan_permute_exp, (void *)&pos1, products.fc_d, products.fc_d, tensors.stream));
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add, (void *)&pos1, products.only1_d, (void *)&neg1, products.fc_d, products.fc_d, tensors.stream));

    // float fs = 1 - fc;
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add, (void *)&pos1, products.only1_d, (void *)&neg1, products.fc_d, products.fs_d, tensors.stream));

    // this->kb1[i] = ((kb1_fst_part * pow(fc, 2)) + (kb1_sec_part * pow(fc, 2) * pow(fs, 2)) + (pow(fs, 2) * kb1s)) * SF;
    HANDLE_CUTENSOR_ERROR(cutensorPermute(tensors.handle, tensors.tensor_plan_permute_log, (void *)&pos2, products.fc_d, products.fcpow_d, tensors.stream));
    HANDLE_CUTENSOR_ERROR(cutensorPermute(tensors.handle, tensors.tensor_plan_permute_exp, (void *)&pos1, products.fcpow_d, products.fcpow_d, tensors.stream));
    HANDLE_CUTENSOR_ERROR(cutensorPermute(tensors.handle, tensors.tensor_plan_permute_log, (void *)&pos2, products.fs_d, products.fspow_d, tensors.stream));
    HANDLE_CUTENSOR_ERROR(cutensorPermute(tensors.handle, tensors.tensor_plan_permute_exp, (void *)&pos1, products.fspow_d, products.fspow_d, tensors.stream));
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_mult, (void *)&pos1, products.kb1_fst_part_d, (void *)&pos1, products.fcpow_d, products.tensor_aux1_d, tensors.stream));
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_mult, (void *)&pos1, products.fcpow_d, (void *)&pos1, products.fspow_d, products.tensor_aux2_d, tensors.stream));
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_mult, (void *)&pos1, products.kb1_sec_part_d, (void *)&pos1, products.tensor_aux2_d, products.tensor_aux2_d, tensors.stream));
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_mult, (void *)&pos1, products.kb1s_d, (void *)&pos1, products.fspow_d, products.kb1_d, tensors.stream));
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add, (void *)&pos1, products.kb1_d, (void *)&pos1, products.tensor_aux2_d, products.kb1_d, tensors.stream));
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add, (void *)&pos1, products.kb1_d, (void *)&pos1, products.tensor_aux1_d, products.kb1_d, tensors.stream));
    HANDLE_CUTENSOR_ERROR(cutensorPermute(tensors.handle, tensors.tensor_plan_permute_id, (void *)&SF, products.kb1_d, products.kb1_d, tensors.stream));
    cudaEventRecord(stop);

    float cuda_time = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuda_time, start, stop);
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
    return "KERNELS,KB1," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string zom_fuction(Products products, Tensor tensors, float A_ZOM, float B_ZOM)
{
    int64_t initial_time, final_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    cudaEventRecord(start);
    float HGHT = 4;
    float CD = 0.01;
    float CR = 0.35;
    float PSICORR = 0.2;
    float CR2 = CR * 2;
    float neg05 = -0.5;
    float pos1 = 1;
    float pos33 = 3.3;
    float negVON = -VON_KARMAN;
    float neg1 = -1;
    // tensor_aux1_d = (CD + CR2 * this->pai_vector[line][col])
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add, (void *)&CD, products.only1_d, (void *)&CR2, products.pai_d, products.tensor_aux1_d, tensors.stream));

    // tensor_aux1_d = pow(tensor_aux1_d, -0.5); ~ exp(−0.5⋅log(tensor_aux1_d))
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_log_mul, (void *)&neg05, products.only1_d, (void *)&pos1, products.tensor_aux1_d, products.tensor_aux1_d, tensors.stream));
    HANDLE_CUTENSOR_ERROR(cutensorPermute(tensors.handle, tensors.tensor_plan_permute_exp, (void *)&pos1, products.tensor_aux1_d, products.tensor_aux1_d, tensors.stream));

    // if (gama < 3.3) gama = 3.3;
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_max, (void *)&pos33, products.only1_d, (void *)&pos1, products.tensor_aux1_d, products.tensor_aux1_d, tensors.stream));

    // tensor_aux1_d = (-VON_KARMAN * gama) + PSICORR
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add, (void *)&negVON, products.tensor_aux1_d, (void *)&PSICORR, products.only1_d, products.tensor_aux1_d, tensors.stream));

    // tensor_aux1_d = pow(exp(1.0), tensor_aux1_d) ~ exp(tensor_aux1_d)
    HANDLE_CUTENSOR_ERROR(cutensorPermute(tensors.handle, tensors.tensor_plan_permute_exp, (void *)&pos1, products.tensor_aux1_d, products.tensor_aux1_d, tensors.stream));

    // tensor_aux2_d = HGHT - this->d0_pointer[pos]
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add, (void *)&HGHT, products.only1_d, (void *)&neg1, products.d0_d, products.tensor_aux2_d, tensors.stream));

    // zom = tensor_aux1_d * tensor_aux2_d
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_mult, (void *)&pos1, products.tensor_aux2_d, (void *)&pos1, products.tensor_aux1_d, products.zom_d, tensors.stream));
    cudaEventRecord(stop);

    float cuda_time = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuda_time, start, stop);
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
    return "KERNELS,ZOM," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string ustar_fuction(Products products, Tensor tensors, float u_const)
{
    int64_t initial_time, final_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    cudaEventRecord(start);
    float zu = 10;
    float pos1 = 1;
    float neg1 = -1;
    float uVON = u_const * VON_KARMAN;
    // tensor_aux1_d = (zu - DISP)
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add, (void *)&zu, products.only1_d, (void *)&neg1, products.d0_d, products.tensor_aux1_d, tensors.stream));

    // tensor_aux1_d = (zu - DISP) / zom
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_div, (void *)&pos1, products.tensor_aux1_d, (void *)&pos1, products.zom_d, products.tensor_aux1_d, tensors.stream));

    // tensor_aux1_d = log((zu - DISP) / zom)
    HANDLE_CUTENSOR_ERROR(cutensorPermute(tensors.handle, tensors.tensor_plan_permute_log, (void *)&pos1, products.tensor_aux1_d, products.tensor_aux1_d, tensors.stream));

    // ustar[i] = (u10 * VON_KARMAN) / log((zu - DISP) / zom);
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_div, (void *)&uVON, products.only1_d, (void *)&pos1, products.tensor_aux1_d, products.ustar_d, tensors.stream));
    cudaEventRecord(stop);

    float cuda_time = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuda_time, start, stop);
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
    return "KERNELS,USTAR," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string aerodynamic_resistance_fuction(Products products, Tensor tensors)
{
    int64_t initial_time, final_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    cudaEventRecord(start);
    float zu = 10.0;
    float pos1 = 1;
    float neg1 = -1;
    // float zoh_terra = zom / pow(exp(1.0), (this->kb1[i]));
    HANDLE_CUTENSOR_ERROR(cutensorPermute(tensors.handle, tensors.tensor_plan_permute_exp, (void *)&pos1, products.kb1_d, products.tensor_aux1_d, tensors.stream));
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_div, (void *)&pos1, products.zom_d, (void *)&pos1, products.tensor_aux1_d, products.tensor_aux1_d, tensors.stream));

    // float temp_kb_1_terra = log(zom / zoh_terra);
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_div, (void *)&pos1, products.zom_d, (void *)&pos1, products.tensor_aux1_d, products.tensor_aux1_d, tensors.stream));
    HANDLE_CUTENSOR_ERROR(cutensorPermute(tensors.handle, tensors.tensor_plan_permute_log, (void *)&pos1, products.kb1_d, products.tensor_aux1_d, tensors.stream));

    // float temp_rah1_terra = (1 / (this->ustar[i] * VON_KARMAN));
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_div, (void *)&pos1, products.only1_d, (void *)&VON_KARMAN, products.ustar_d, products.tensor_aux2_d, tensors.stream));

    // float temp_rah2 = log(((zu - DISP) / zom));
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add, (void *)&zu, products.only1_d, (void *)&neg1, products.d0_d, products.tensor_aux3_d, tensors.stream));
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_div, (void *)&pos1, products.tensor_aux3_d, (void *)&pos1, products.zom_d, products.tensor_aux3_d, tensors.stream));
    HANDLE_CUTENSOR_ERROR(cutensorPermute(tensors.handle, tensors.tensor_plan_permute_log, (void *)&pos1, products.tensor_aux3_d, products.tensor_aux3_d, tensors.stream));

    // float temp_rah3_terra = temp_rah1_terra * temp_kb_1_terra;
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_mult, (void *)&pos1, products.tensor_aux2_d, (void *)&pos1, products.tensor_aux1_d, products.tensor_aux1_d, tensors.stream));

    // this->aerodynamic_resistance[i] = temp_rah1_terra * temp_rah2 + temp_rah3_terra;
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_mult, (void *)&pos1, products.tensor_aux2_d, (void *)&pos1, products.tensor_aux3_d, products.rah_d, tensors.stream));
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add, (void *)&pos1, products.rah_d, (void *)&pos1, products.tensor_aux1_d, products.rah_d, tensors.stream));
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
        rah_correction_cycle_STEEP<<<blocks_n, threads_n>>>(products.net_radiation_d, products.soil_heat_d, products.ndvi_d, products.surface_temperature_d, products.d0_d, products.kb1_d, products.zom_d, products.ustar_d, products.rah_d, products.sensible_heat_flux_d, products.a_d, products.b_d, ndvi_max, ndvi_min);
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
        rah_correction_cycle_ASEBAL<<<blocks_n, threads_n>>>(products.net_radiation_d, products.soil_heat_d, products.ndvi_d, products.surface_temperature_d, products.kb1_d, products.zom_d, products.ustar_d, products.rah_d, products.sensible_heat_flux_d, products.a_d, products.b_d, u200, products.stop_condition_d);
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

string sensible_heat_flux_function(Products products, Tensor tensors)
{
    int64_t initial_time, final_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    cudaEventRecord(start);
    float pos1 = 1;
    float neg1 = -1;
    float neg27315 = -273.15;
    float RHO_AIR = RHO * SPECIFIC_HEAT_AIR;
    // tensor_aux1_d = this->surface_temperature[i] - 273.15
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add,
                                                            (void *)&pos1, products.surface_temperature_d, (void *)&neg27315, products.only1_d, products.tensor_aux1_d, tensors.stream));

    // tensor_aux1_d = a + b * (this->surface_temperature[i] - 273.15))
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add,
                                                            (void *)&products.a_d, products.only1_d, (void *)&products.b_d, products.tensor_aux1_d, products.tensor_aux1_d, tensors.stream));

    // sensible_heat_flux[i] = RHO * SPECIFIC_HEAT_AIR * (a + b * (this->surface_temperature[i] - 273.15)) / this->aerodynamic_resistance[i];
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_div,
                                                            (void *)&RHO_AIR, products.tensor_aux1_d, (void *)&pos1, products.rah_d, products.sensible_heat_flux_d, tensors.stream));

    // tensor_aux2_d = this->net_radiation[i] - this->soil_heat[i]
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add,
                                                            (void *)&pos1, products.net_radiation_d, (void *)&neg1, products.soil_heat_d, products.tensor_aux2_d, tensors.stream));

    // if (sensible_heat_flux[i] > (this->net_radiation[i] - this->soil_heat[i])) sensible_heat_flux[i] = this->net_radiation[i] - this->soil_heat[i];
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_min,
                                                            (void *)&pos1, products.sensible_heat_flux_d, (void *)&pos1, products.tensor_aux2_d, products.sensible_heat_flux_d, tensors.stream));
    cudaEventRecord(stop);

    float cuda_time = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuda_time, start, stop);
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
    return "KERNELS,SENSIBLE_HEAT_FLUX," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::converge_rah_cycle(Products products, Station station, Tensor tensors)
{
    string result = "";
    int64_t initial_time, final_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    cudaEventRecord(start);
    float ustar_station = (VON_KARMAN * station.v6) / (log(station.WIND_SPEED / station.SURFACE_ROUGHNESS));
    float u10 = (ustar_station / VON_KARMAN) * log(10 / station.SURFACE_ROUGHNESS);
    float u200 = (ustar_station / VON_KARMAN) * log(200 / station.SURFACE_ROUGHNESS);

    float ndvi_min = thrust::reduce(thrust::device,
                                    products.ndvi_d,
                                    products.ndvi_d + products.height_band * products.width_band,
                                    1.0f, // Initial value
                                    thrust::minimum<float>());

    float ndvi_max = thrust::reduce(thrust::device,
                                    products.ndvi_d,
                                    products.ndvi_d + products.height_band * products.width_band,
                                    -1.0f, // Initial value
                                    thrust::maximum<float>());

    if (model_method == 0) { // STEEP
        result += d0_fuction(products, tensors);
        result += zom_fuction(products, tensors, station.A_ZOM, station.B_ZOM);
        result += ustar_fuction(products, tensors, u10);
        result += kb_function(products, tensors, ndvi_max, ndvi_min);
        result += aerodynamic_resistance_fuction(products, tensors);
        result += rah_correction_function_blocks_STEEP(products, ndvi_min, ndvi_max);
        result += sensible_heat_flux_function(products, tensors);
    } else { // ASEBAL
        result += zom_fuction(products, tensors, station.A_ZOM, station.B_ZOM);
        result += ustar_fuction(products, tensors, u200);
        result += aerodynamic_resistance_fuction(products, tensors);
        result += rah_correction_function_blocks_ASEBAL(products, u200);
        result += sensible_heat_flux_function(products, tensors);
    }
    cudaEventRecord(stop);

    float cuda_time = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuda_time, start, stop);
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
    result += "KERNELS,P3_RAH," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
    return result;
};
