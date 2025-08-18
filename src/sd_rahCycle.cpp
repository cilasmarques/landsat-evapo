#include "sensors.h"
#include "surfaceData.h"
#include "thread_utils.h" // Include for parallel_for
#include <atomic> // Required for std::atomic with global_a and global_b if strict C++11 is not used. Otherwise, ensure proper synchronization.

// Note: global_a and global_b are modified by rah_correction functions and read by sensible_heat_flux_function.
// The current parallelization strategy assumes that rah_correction functions complete their updates to global_a and global_b
// *before* sensible_heat_flux_function is called or parallelized.
// If rah_correction's internal loops are parallelized, ensure global_a/global_b updates are handled correctly (e.g., not in parallel part or synchronized).
std::atomic<float> global_a(0);
std::atomic<float> global_b(0);


// Worker for d0_function
void d0_function_worker(Products products, float CD1, float HGHT, int start_row, int end_row) {
    for (int r = start_row; r < end_row; ++r) {
        for (int c = 0; c < products.width_band; ++c) {
            int i = r * products.width_band + c;
            float cd1_pai_root = sqrt(CD1 * products.pai[i]);
            products.d0[i] = HGHT * ((1.0 - (1.0 / cd1_pai_root)) + (pow(exp(1.0), -cd1_pai_root) / cd1_pai_root));
        }
    }
}

string d0_fuction(Products products)
{
    float CD1 = 20.6;
    float HGHT = 4.0;
    int64_t initial_time, final_time;
    system_clock::time_point begin, end;
    float general_time;

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
    begin = system_clock::now();

    parallel_for(products.height_band, [&](int start_row, int end_row) {
        d0_function_worker(products, CD1, HGHT, start_row, end_row);
    });

    end = system_clock::now();
    general_time = duration_cast<nanoseconds>(end - begin).count() / 1000000.0;
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
    return "PARALLEL,D0," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

// Worker for kb_function
void kb_function_worker(Products products, float ndvi_max, float ndvi_min, int start_row, int end_row) {
    float HGHT = 4.0;
    float VON_KARMAN = 0.41;
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
    float soil_moisture_day_rel = 0.33;

    for (int r = start_row; r < end_row; ++r) {
        for (int c = 0; c < products.width_band; ++c) {
            int i = r * products.width_band + c;
            float fc = 1.0 - pow((products.ndvi[i] - ndvi_max) / (ndvi_min - ndvi_max), 0.4631);
            float fs = 1.0 - fc;

            float Re = (products.ustar[i] * 0.009) / visc;
            float Ct = pow(pr, -0.667) * pow(Re, -0.5);
            float ratio = c1 - c2 * (exp(cd * -c3 * products.pai[i]));
            float nec = (cd * products.pai[i]) / (ratio * ratio * 2.0);
            float kbs = 2.46 * pow(Re, 0.25) - 2.0;

            float kb1_fst_part = (cd * VON_KARMAN) / (4.0 * ct * ratio * (1.0 - exp(nec * -0.5)));
            float kb1_sec_part = pow(fc, 2.0) + (VON_KARMAN * ratio * (products.zom[i] / HGHT) / Ct);
            float kb1_trd_part = pow(fc, 2.0) * pow(fs, 2.0) + kbs * pow(fs, 2.0);
            float kb_ini = kb1_fst_part * kb1_sec_part * kb1_trd_part;

            float SF = sf_c + (1.0 / (1.0 + exp(sf_d - sf_e * soil_moisture_day_rel)));

            products.kb1[i] = kb_ini * SF;
        }
    }
}

string kb_function(Products products, float ndvi_max, float ndvi_min)
{
    int64_t initial_time, final_time;
    system_clock::time_point begin, end;
    float general_time;

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
    begin = system_clock::now();

    parallel_for(products.height_band, [&](int start_row, int end_row) {
        kb_function_worker(products, ndvi_max, ndvi_min, start_row, end_row);
    });

    end = system_clock::now();
    general_time = duration_cast<nanoseconds>(end - begin).count() / 1000000.0;
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
    return "PARALLEL,KB1," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

// Worker for zom_function (STEEP)
void zom_function_steep_worker(Products products, float HGHT, float CD, float CR, float PSICORR, int start_row, int end_row) {
    for (int r = start_row; r < end_row; ++r) {
        for (int c = 0; c < products.width_band; ++c) {
            int i = r * products.width_band + c;
            float gama = pow((CD + CR * (products.pai[i] / 2.0)), -0.5);
            if (gama < 3.3)
                gama = 3.3;
            products.zom[i] = (HGHT - products.d0[i]) * exp(-VON_KARMAN * gama) + PSICORR;
        }
    }
}

// Worker for zom_function (ASEBAL)
void zom_function_asebal_worker(Products products, float A_ZOM, float B_ZOM, int start_row, int end_row) {
    for (int r = start_row; r < end_row; ++r) {
        for (int c = 0; c < products.width_band; ++c) {
            int i = r * products.width_band + c;
            products.zom[i] = exp((A_ZOM * products.ndvi[i] / products.albedo[i]) + B_ZOM);
        }
    }
}

string zom_fuction(Products products, float A_ZOM, float B_ZOM)
{
    int64_t initial_time, final_time;
    system_clock::time_point begin, end;
    float general_time;

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
    begin = system_clock::now();

    if (model_method == 0) // STEEP
    {
        float HGHT = 4.0;
        float CD = 0.01;
        float CR = 0.35;
        float PSICORR = 0.2;
        parallel_for(products.height_band, [&](int start_row, int end_row) {
            zom_function_steep_worker(products, HGHT, CD, CR, PSICORR, start_row, end_row);
        });
    }
    else // ASEBAL
    {
        parallel_for(products.height_band, [&](int start_row, int end_row) {
            zom_function_asebal_worker(products, A_ZOM, B_ZOM, start_row, end_row);
        });
    }

    end = system_clock::now();
    general_time = duration_cast<nanoseconds>(end - begin).count() / 1000000.0;
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
    return "PARALLEL,ZOM," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

// Worker for ustar_function (STEEP)
void ustar_function_steep_worker(Products products, float u_const, int start_row, int end_row) {
    for (int r = start_row; r < end_row; ++r) {
        for (int c = 0; c < products.width_band; ++c) {
            int i = r * products.width_band + c;
            products.ustar[i] = (u_const * VON_KARMAN) / log((10.0 - products.d0[i]) / products.zom[i]);
        }
    }
}

// Worker for ustar_function (ASEBAL)
void ustar_function_asebal_worker(Products products, float u_const, int start_row, int end_row) {
    for (int r = start_row; r < end_row; ++r) {
        for (int c = 0; c < products.width_band; ++c) {
            int i = r * products.width_band + c;
            products.ustar[i] = (u_const * VON_KARMAN) / log(200.0 / products.zom[i]);
        }
    }
}

string ustar_fuction(Products products, float u_const)
{
    int64_t initial_time, final_time;
    system_clock::time_point begin, end;
    float general_time;

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
    begin = system_clock::now();

    if (model_method == 0) // STEEP
    {
        parallel_for(products.height_band, [&](int start_row, int end_row) {
            ustar_function_steep_worker(products, u_const, start_row, end_row);
        });
    }
    else // ASEBAL
    {
        parallel_for(products.height_band, [&](int start_row, int end_row) {
            ustar_function_asebal_worker(products, u_const, start_row, end_row);
        });
    }

    end = system_clock::now();
    general_time = duration_cast<nanoseconds>(end - begin).count() / 1000000.0;
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
    return "PARALLEL,USTAR," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

// Worker for aerodynamic_resistance_fuction (STEEP)
void aerodynamic_resistance_steep_worker(Products products, int start_row, int end_row) {
    for (int r = start_row; r < end_row; ++r) {
        for (int c = 0; c < products.width_band; ++c) {
            int i = r * products.width_band + c;
            float rah_fst_part = 1.0 / (products.ustar[i] * VON_KARMAN);
            float rah_sec_part = log((10.0 - products.d0[i]) / products.zom[i]);
            float rah_trd_part = rah_fst_part * products.kb1[i];
            products.aerodynamic_resistance[i] = (rah_fst_part * rah_sec_part) + rah_trd_part;
        }
    }
}

// Worker for aerodynamic_resistance_fuction (ASEBAL)
void aerodynamic_resistance_asebal_worker(Products products, int start_row, int end_row) {
    for (int r = start_row; r < end_row; ++r) {
        for (int c = 0; c < products.width_band; ++c) {
            int i = r * products.width_band + c;
            products.aerodynamic_resistance[i] = log(2.0 / 0.1) / (products.ustar[i] * VON_KARMAN);
        }
    }
}

string aerodynamic_resistance_fuction(Products products)
{
    int64_t initial_time, final_time;
    system_clock::time_point begin, end;
    float general_time;

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
    begin = system_clock::now();

    if (model_method == 0) // STEEP
    {
        parallel_for(products.height_band, [&](int start_row, int end_row) {
            aerodynamic_resistance_steep_worker(products, start_row, end_row);
        });
    }
    else // ASEBAL
    {
        parallel_for(products.height_band, [&](int start_row, int end_row) {
            aerodynamic_resistance_asebal_worker(products, start_row, end_row);
        });
    }

    end = system_clock::now();
    general_time = duration_cast<nanoseconds>(end - begin).count() / 1000000.0;
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
    return "PARALLEL,RAH_INI," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

// Worker for the inner loop of rah_correction_function_STEEP
void rah_correction_steep_inner_loop_worker(Products products, float a_val, float b_val, int start_row, int end_row) {
    for (int r = start_row; r < end_row; ++r) {
        for (int c = 0; c < products.width_band; ++c) {
            int i = r * products.width_band + c;
            float dt_final = a_val + b_val * (products.surface_temperature[i]);
            products.sensible_heat_flux[i] = RHO * SPECIFIC_HEAT_AIR * dt_final / products.aerodynamic_resistance[i];
            float L = -1.0 * ((RHO * SPECIFIC_HEAT_AIR * pow(products.ustar[i], 3.0) * products.surface_temperature[i]) / (VON_KARMAN * GRAVITY * products.sensible_heat_flux[i]));

            float y2 = pow((1.0 - (16.0 * (10.0 - products.d0[i])) / L), 0.25);
            float x200 = pow((1.0 - (16.0 * (10.0 - products.d0[i])) / L), 0.25);

            float psi2, psi200;
            if (!isnan(L) && L > 0)
            {
                psi2 = -5.0 * ((10.0 - products.d0[i]) / L);
                psi200 = -5 * ((10 - products.d0[i]) / L);
            }
            else
            {
                psi2 = 2.0 * log((1.0 + y2 * y2) / 2.0);
                psi200 = 2.0 * log((1.0 + x200) / 2.0) + log((1.0 + x200 * x200) / 2.0) - 2.0 * atan(x200) + 0.5 * PI;
            }

            products.ustar[i] = (VON_KARMAN * products.ustar[i]) / (log((10.0 - products.d0[i]) / products.zom[i]) - psi200);

            float rah_fst_part = 1.0 / (products.ustar[i] * VON_KARMAN);
            float rah_sec_part = log((10.0 - products.d0[i]) / products.zom[i]) - psi2;
            float rah_trd_part = rah_fst_part * products.kb1[i];
            products.aerodynamic_resistance[i] = (rah_fst_part * rah_sec_part) + rah_trd_part;
        }
    }
}


string rah_correction_function_STEEP(Products products, float ndvi_min, float ndvi_max)
{
    system_clock::time_point begin, end;
    int64_t initial_time, final_time;
    float general_time;

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
    begin = system_clock::now();

    for (int iter = 0; iter < 2; iter++) // Renamed 'i' to 'iter' to avoid conflict with pixel index
    {
        int hot_pos = products.hotEndmemberPos[0] * products.width_band + products.hotEndmemberPos[1];
        int cold_pos = products.coldEndmemberPos[0] * products.width_band + products.coldEndmemberPos[1];

        float rah_ini_hot = products.aerodynamic_resistance[hot_pos];
        float rah_ini_cold = products.aerodynamic_resistance[cold_pos];

        float fc_hot = 1.0 - pow((products.ndvi[hot_pos] - ndvi_max) / (ndvi_min - ndvi_max), 0.4631);
        float fc_cold = 1.0 - pow((products.ndvi[cold_pos] - ndvi_max) / (ndvi_min - ndvi_max), 0.4631);

        float LE_hot = 0.55 * fc_hot * (products.net_radiation[hot_pos] - products.soil_heat[hot_pos]) * 0.78;
        float LE_cold = 1.75 * fc_cold * (products.net_radiation[cold_pos] - products.soil_heat[cold_pos]) * 0.78;

        float H_cold = products.net_radiation[cold_pos] - products.soil_heat[cold_pos] - LE_hot;
        float dt_cold = H_cold * rah_ini_cold / (RHO * SPECIFIC_HEAT_AIR);

        float H_hot = products.net_radiation[hot_pos] - products.soil_heat[hot_pos] - LE_cold;
        float dt_hot = H_hot * rah_ini_hot / (RHO * SPECIFIC_HEAT_AIR);

        float b_val = (dt_hot - dt_cold) / (products.surface_temperature[hot_pos] - products.surface_temperature[cold_pos]);
        float a_val = dt_cold - (b_val * products.surface_temperature[cold_pos]);

        global_a.store(a_val); // Store atomically
        global_b.store(b_val); // Store atomically

        parallel_for(products.height_band, [&](int start_row, int end_row) {
            rah_correction_steep_inner_loop_worker(products, a_val, b_val, start_row, end_row);
        });
    }

    end = system_clock::now();
    general_time = duration_cast<nanoseconds>(end - begin).count() / 1000000.0;
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
    return "PARALLEL,RAH_CYCLE," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
}

// Worker for the inner loop of rah_correction_function_ASEBAL
void rah_correction_asebal_inner_loop_worker(Products products, float a_val, float b_val, float u200_val, int start_row, int end_row) {
    for (int r = start_row; r < end_row; ++r) {
        for (int c = 0; c < products.width_band; ++c) {
            int i = r * products.width_band + c;
            float dt_final = a_val + b_val * products.surface_temperature[i];
            products.sensible_heat_flux[i] = RHO * SPECIFIC_HEAT_AIR * dt_final / products.aerodynamic_resistance[i];
            float L = -1.0 * ((RHO * SPECIFIC_HEAT_AIR * pow(products.ustar[i], 3.0) * products.surface_temperature[i]) / (VON_KARMAN * GRAVITY * products.sensible_heat_flux[i]));

            float x1 = pow((1.0 - (16.0 * 0.1) / L), 0.25);
            float x2 = pow((1.0 - (16.0 * 2.0) / L), 0.25);
            float x200_L = pow((1.0 - (16.0 * 200.0) / L), 0.25); // Renamed to avoid conflict

            float psi1, psi2, psi200_val; // Renamed to avoid conflict
            if (!isnan(L) && L > 0)
            {
                psi1 = -5.0 * (0.1 / L);
                psi2 = -5.0 * (2.0 / L);
                psi200_val = -5.0 * (2.0 / L);
            }
            else
            {
                psi1 = 2.0 * log((1.0 + x1 * x1) / 2.0);
                psi2 = 2.0 * log((1.0 + x2 * x2) / 2.0);
                psi200_val = 2.0 * log((1.0 + x200_L) / 2.0) + log((1.0 + x200_L * x200_L) / 2.0) - 2.0 * atan(x200_L) + 0.5 * PI;
            }

            products.ustar[i] = (VON_KARMAN * u200_val) / (log(200.0 / products.zom[i]) - psi200_val);
            products.aerodynamic_resistance[i] = (log(2.0 / 0.1) - psi2 + psi1) / (products.ustar[i] * VON_KARMAN);
        }
    }
}

string rah_correction_function_ASEBAL(Products products, float u200)
{
    system_clock::time_point begin, end;
    int64_t initial_time, final_time;
    float general_time;

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
    begin = system_clock::now();

    int iter_count = 0; // Renamed 'i' to 'iter_count'
    while (true)
    {
        int hot_pos = products.hotEndmemberPos[0] * products.width_band + products.hotEndmemberPos[1];
        int cold_pos = products.coldEndmemberPos[0] * products.width_band + products.coldEndmemberPos[1];

        float rah_ini_hot = products.aerodynamic_resistance[hot_pos];
        // float rah_ini_cold = products.aerodynamic_resistance[cold_pos]; // Not used directly in a_val, b_val calculation here

        float H_cold = products.net_radiation[cold_pos] - products.soil_heat[cold_pos];
        float dt_cold = H_cold * products.aerodynamic_resistance[cold_pos] / (RHO * SPECIFIC_HEAT_AIR); // Used rah_ini_cold equivalent

        float H_hot = products.net_radiation[hot_pos] - products.soil_heat[hot_pos];
        float dt_hot = H_hot * rah_ini_hot / (RHO * SPECIFIC_HEAT_AIR);

        float b_val = (dt_hot - dt_cold) / (products.surface_temperature[hot_pos] - products.surface_temperature[cold_pos]);
        float a_val = dt_cold - (b_val * (products.surface_temperature[cold_pos]));

        global_a.store(a_val); // Store atomically
        global_b.store(b_val); // Store atomically

        parallel_for(products.height_band, [&](int start_row, int end_row) {
            rah_correction_asebal_inner_loop_worker(products, a_val, b_val, u200, start_row, end_row);
        });

        if ((iter_count > 0) && (fabs(1 - (rah_ini_hot / products.aerodynamic_resistance[hot_pos])) < 0.05))
            break;
        else
            iter_count++;
    }

    end = system_clock::now();
    general_time = duration_cast<nanoseconds>(end - begin).count() / 1000000.0;
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
    return "PARALLEL,RAH_CYCLE," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
}

// Worker for sensible_heat_flux_function
void sensible_heat_flux_function_worker(Products products, float current_global_a, float current_global_b, int start_row, int end_row) {
    for (int r = start_row; r < end_row; ++r) {
        for (int c = 0; c < products.width_band; ++c) {
            int i = r * products.width_band + c;
            products.sensible_heat_flux[i] = RHO * SPECIFIC_HEAT_AIR * (current_global_a + current_global_b * (products.surface_temperature[i])) / products.aerodynamic_resistance[i];
            if (!isnan(products.sensible_heat_flux[i]) && products.sensible_heat_flux[i] > (products.net_radiation[i] - products.soil_heat[i]))
                products.sensible_heat_flux[i] = products.net_radiation[i] - products.soil_heat[i];
        }
    }
}

string sensible_heat_flux_function(Products products)
{
    int64_t initial_time, final_time;
    system_clock::time_point begin, end;
    float general_time;

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
    begin = system_clock::now();

    // Load atomic globals once before parallel section
    float current_a = global_a.load();
    float current_b = global_b.load();

    parallel_for(products.height_band, [&](int start_row, int end_row) {
        sensible_heat_flux_function_worker(products, current_a, current_b, start_row, end_row);
    });

    end = system_clock::now();
    general_time = duration_cast<nanoseconds>(end - begin).count() / 1000000.0;
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
    return "PARALLEL,SENSIBLE_HEAT_FLUX," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::converge_rah_cycle(Products products, Station station)
{
    string result = "";
    int64_t initial_time, final_time;
    system_clock::time_point begin, end;
    float general_time;

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
    begin = system_clock::now();

    float ustar_station = (VON_KARMAN * station.v6) / (log(station.WIND_SPEED / station.SURFACE_ROUGHNESS));
    float u10 = (ustar_station / VON_KARMAN) * log(10.0 / station.SURFACE_ROUGHNESS);
    float u200 = (ustar_station / VON_KARMAN) * log(200.0 / station.SURFACE_ROUGHNESS);

    // This loop for ndvi_min/max is sequential and fast, no need to parallelize.
    float ndvi_min = 1.0;
    float ndvi_max = -1.0;
    for (int i = 0; i < this->height_band * this->width_band; i++)
    {
        if (products.ndvi[i] < ndvi_min)
            ndvi_min = products.ndvi[i];
        if (products.ndvi[i] > ndvi_max)
            ndvi_max = products.ndvi[i];
    }

    if (model_method == 0)
    { // STEEP
        result += d0_fuction(products);
        result += zom_fuction(products, station.A_ZOM, station.B_ZOM);
        result += ustar_fuction(products, u10);
        result += kb_function(products, ndvi_max, ndvi_min);
        result += aerodynamic_resistance_fuction(products);
        // rah_correction_function_STEEP is now internally parallelized for its main pixel loop
        result += rah_correction_function_STEEP(products, ndvi_min, ndvi_max);
        result += sensible_heat_flux_function(products); // Internally parallelized
    }
    else
    { // ASEBAL
        result += zom_fuction(products, station.A_ZOM, station.B_ZOM);
        result += ustar_fuction(products, u200);
        result += aerodynamic_resistance_fuction(products);
        // rah_correction_function_ASEBAL is now internally parallelized for its main pixel loop
        result += rah_correction_function_ASEBAL(products, u200);
        result += sensible_heat_flux_function(products); // Internally parallelized
    }

    end = system_clock::now();
    general_time = duration_cast<nanoseconds>(end - begin).count() / 1000000.0;
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
    result += "PARALLEL,P3_RAH," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
    return result;
};