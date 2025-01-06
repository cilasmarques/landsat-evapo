#include "sensors.h"
#include "surfaceData.h"

float global_a = 0;
float global_b = 0;

string d0_fuction(Products products)
{
    float CD1 = 20.6;
    float HGHT = 4;

    int64_t initial_time, final_time;
    system_clock::time_point begin, end;
    float general_time;

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    begin = system_clock::now();
    for (int i = 0; i < products.height_band * products.width_band; i++) {
        float cd1_pai_root = sqrt(CD1 * products.pai[i]);

        products.d0[i] = HGHT * ((1 - (1 / cd1_pai_root)) + (pow(exp(1.0), -cd1_pai_root) / cd1_pai_root));
    }
    end = system_clock::now();

    general_time = duration_cast<nanoseconds>(end - begin).count() / 1000000.0;
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
    return "SERIAL,D0," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string kb_function(Products products, float ndvi_max, float ndvi_min)
{
    int64_t initial_time, final_time;
    system_clock::time_point begin, end;
    float general_time;

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    begin = system_clock::now();
    float HGHT = 4;
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

    for (int i = 0; i < products.height_band * products.width_band; i++) {
        float Re_star = (products.ustar[i] * 0.009) / visc;
        float Ct_star = pow(pr, -0.667) * pow(Re_star, -0.5);
        float beta = c1 - c2 * (exp((cd * -c3 * products.pai[i])));
        float nec_terra = (cd * products.pai[i]) / (beta * beta * 2);

        float kb1_fst_part = (cd * VON_KARMAN) / (4 * ct * beta * (1 - exp(nec_terra * -0.5)));
        float kb1_sec_part = (beta * VON_KARMAN * (products.zom[i] / HGHT)) / Ct_star;
        float kb1s = (pow(Re_star, 0.25) * 2.46) - 2;

        float fc = 1 - pow((products.ndvi[i] - ndvi_max) / (ndvi_min - ndvi_max), 0.4631);
        float fs = 1 - fc;

        float SF = sf_c + (1 / (1 + pow(exp(1.0), (sf_d - (sf_e * soil_moisture_day_rel)))));

        products.kb1[i] = ((kb1_fst_part * pow(fc, 2)) + (kb1_sec_part * pow(fc, 2) * pow(fs, 2)) + (pow(fs, 2) * kb1s)) * SF;
    }
    end = system_clock::now();

    general_time = duration_cast<nanoseconds>(end - begin).count() / 1000000.0;
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
    return "SERIAL,KB1," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string zom_fuction(Products products, float A_ZOM, float B_ZOM)
{
    int64_t initial_time, final_time;
    system_clock::time_point begin, end;
    float general_time;

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    begin = system_clock::now();
    if (model_method == 0) {
        float HGHT = 4;
        float CD = 0.01;
        float CR = 0.35;
        float PSICORR = 0.2;

        for (int i = 0; i < products.height_band * products.width_band; i++) {
            float gama = pow((CD + CR * (products.pai[i] / 2)), -0.5);
            if (gama < 3.3)
                gama = 3.3;

            products.zom[i] = (HGHT - products.d0[i]) * pow(exp(1.0), (-VON_KARMAN * gama) + PSICORR);
        }
    } else {
        for (int i = 0; i < products.height_band * products.width_band; i++)
            products.zom[i] = exp(A_ZOM + B_ZOM * products.ndvi[i]);
    }
    end = system_clock::now();

    general_time = duration_cast<nanoseconds>(end - begin).count() / 1000000.0;
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
    return "SERIAL,ZOM," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string ustar_fuction(Products products, float u_const)
{
    int64_t initial_time, final_time;
    system_clock::time_point begin, end;
    float general_time;

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    begin = system_clock::now();
    if (model_method == 0) {
        for (int i = 0; i < products.height_band * products.width_band; i++)
            products.ustar[i] = (u_const * VON_KARMAN) / logf((10 - products.d0[i]) / products.zom[i]);
    } else {
        for (int i = 0; i < products.height_band * products.width_band; i++)
            products.ustar[i] = (u_const * VON_KARMAN) / logf(200 / products.zom[i]);
    }
    end = system_clock::now();

    general_time = duration_cast<nanoseconds>(end - begin).count() / 1000000.0;
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
    return "SERIAL,USTAR," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string aerodynamic_resistance_fuction(Products products)
{
    int64_t initial_time, final_time;
    system_clock::time_point begin, end;
    float general_time;

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    begin = system_clock::now();
    if (model_method == 0) {
        for (int i = 0; i < products.height_band * products.width_band; i++) {
            float zoh_terra = products.zom[i] / pow(exp(1.0), (products.kb1[i]));
            float temp_rah1_corr_terra = 1 / (products.ustar[i] * VON_KARMAN);
            float temp_rah2_corr_terra = logf((10 - products.d0[i]) / products.zom[i]);
            float temp_rah3_corr_terra = temp_rah1_corr_terra * logf(products.zom[i] / zoh_terra);
            products.aerodynamic_resistance[i] = (temp_rah1_corr_terra * temp_rah2_corr_terra) + temp_rah3_corr_terra;
        }
    } else {
        for (int i = 0; i < products.height_band * products.width_band; i++)
            products.aerodynamic_resistance[i] = logf(2.0 / 0.1) / (products.ustar[i] * VON_KARMAN);
    }
    end = system_clock::now();

    general_time = duration_cast<nanoseconds>(end - begin).count() / 1000000.0;
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
    return "SERIAL,RAH_INI," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string rah_correction_function_blocks_STEEP(Products products, float ndvi_min, float ndvi_max)
{
    system_clock::time_point begin, end;
    int64_t initial_time, final_time;
    float general_time;

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    begin = system_clock::now();
    for (int i = 0; i < 2; i++) {
        int hot_pos = products.hotEndmemberPos[0] * products.width_band + products.hotEndmemberPos[1];
        int cold_pos = products.coldEndmemberPos[0] * products.width_band + products.coldEndmemberPos[1];

        float rah_ini_pq_terra = products.aerodynamic_resistance[hot_pos];
        float rah_ini_pf_terra = products.aerodynamic_resistance[cold_pos];

        float fc_hot = 1 - pow((products.ndvi[hot_pos] - ndvi_max) / (ndvi_min - ndvi_max), 0.4631);
        float fc_cold = 1 - pow((products.ndvi[cold_pos] - ndvi_max) / (ndvi_min - ndvi_max), 0.4631);

        float LEc_terra = 0.55 * fc_hot * (products.net_radiation[hot_pos] - products.soil_heat[hot_pos]) * 0.78;
        float LEc_terra_pf = 1.75 * fc_cold * (products.net_radiation[cold_pos] - products.soil_heat[cold_pos]) * 0.78;

        float H_pf_terra = products.net_radiation[cold_pos] - products.soil_heat[cold_pos];
        float dt_pf_terra = H_pf_terra * rah_ini_pf_terra / (RHO * SPECIFIC_HEAT_AIR);

        float H_pq_terra = products.net_radiation[hot_pos] - products.soil_heat[hot_pos];
        float dt_pq_terra = H_pq_terra * rah_ini_pq_terra / (RHO * SPECIFIC_HEAT_AIR);

        float b = (dt_pq_terra - dt_pf_terra) / (products.surface_temperature[hot_pos] - products.surface_temperature[cold_pos]);
        float a = dt_pf_terra - (b * products.surface_temperature[cold_pos]);

        global_a = a;
        global_b = b;

        for (int i = 0; i < products.height_band * products.width_band; i++) {
            float dT_ini_terra = a + b * (products.surface_temperature[i]);

            float sensibleHeatFlux = RHO * SPECIFIC_HEAT_AIR * (dT_ini_terra) / products.aerodynamic_resistance[i];
            float L = -1 * ((RHO * SPECIFIC_HEAT_AIR * pow(products.ustar[i], 3) * products.surface_temperature[i]) / (VON_KARMAN * GRAVITY * sensibleHeatFlux));

            float DISP = products.d0[i];
            float y2 = pow((1 - (16 * (10 - DISP)) / L), 0.25);
            float x200 = pow((1 - (16 * (10 - DISP)) / L), 0.25);

            float psi2, psi200;
            if (!isnan(L) && L > 0) {
                psi2 = -5 * ((10 - DISP) / L);
                psi200 = -5 * ((10 - DISP) / L);
            } else {
                psi2 = 2 * logf((1 + y2 * y2) / 2);
                psi200 = 2 * logf((1 + x200) / 2) + logf((1 + x200 * x200) / 2) - 2 * atan(x200) + 0.5 * M_PI;
            }

            float ust = (VON_KARMAN * products.ustar[i]) / (logf((10 - DISP) / products.zom[i]) - psi200);

            float zoh_terra = products.zom[i] / pow(exp(1.0), (products.kb1[i] + psi200));
            float temp_rah1_corr_terra = 1 / (products.ustar[i] * VON_KARMAN);
            float temp_rah2_corr_terra = logf((10 - DISP) / products.zom[i]) - psi2;
            float temp_rah3_corr_terra = temp_rah1_corr_terra * logf(products.zom[i] / zoh_terra);
            float rah = (temp_rah1_corr_terra * temp_rah2_corr_terra) + temp_rah3_corr_terra;

            products.ustar[i] = ust;
            products.aerodynamic_resistance[i] = rah;
            products.sensible_heat_flux[i] = sensibleHeatFlux;
        }
    }
    end = system_clock::now();

    general_time = duration_cast<nanoseconds>(end - begin).count() / 1000000.0;
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
    return "SERIAL,RAH_CYCLE," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
}

string rah_correction_function_blocks_ASEBAL(Products products, float ndvi_min, float ndvi_max, float u200)
{
    system_clock::time_point begin, end;
    int64_t initial_time, final_time;
    float general_time;

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    begin = system_clock::now();
    int i = 0;
    while (true) {
        int hot_pos = products.hotEndmemberPos[0] * products.width_band + products.hotEndmemberPos[1];
        int cold_pos = products.coldEndmemberPos[0] * products.width_band + products.coldEndmemberPos[1];

        float rah_ini_pq_terra = products.aerodynamic_resistance[hot_pos];
        float rah_ini_pf_terra = products.aerodynamic_resistance[cold_pos];

        float H_pf_terra = products.net_radiation[cold_pos] - products.soil_heat[cold_pos];
        float dt_pf_terra = H_pf_terra * rah_ini_pf_terra / (RHO * SPECIFIC_HEAT_AIR);

        float H_pq_terra = products.net_radiation[hot_pos] - products.soil_heat[hot_pos];
        float dt_pq_terra = H_pq_terra * rah_ini_pq_terra / (RHO * SPECIFIC_HEAT_AIR);

        int posh = 8 * products.width_band + 22;
        int posc = 13 * products.width_band + 42;

        float b = (dt_pq_terra - dt_pf_terra) / (products.surface_temperature[hot_pos] - products.surface_temperature[cold_pos]);
        float a = dt_pf_terra - (b * (products.surface_temperature[cold_pos]));

        global_a = a;
        global_b = b;

        for (int i = 0; i < products.height_band * products.width_band; i++) {
            float dT_ini_terra = a + b * (products.surface_temperature[i]);

            float sensibleHeatFlux = RHO * SPECIFIC_HEAT_AIR * (dT_ini_terra) / products.aerodynamic_resistance[i];
            float L = -1 * ((RHO * SPECIFIC_HEAT_AIR * pow(products.ustar[i], 3) * products.surface_temperature[i]) / (VON_KARMAN * GRAVITY * sensibleHeatFlux));

            float x1 = pow((1 - (16 * 0.1) / L), 0.25);
            float x2 = pow((1 - (16 * 2) / L), 0.25);
            float x200 = pow((1 - (16 * 200) / L), 0.25);

            float psi1, psi2, psi200;
            if (!isnan(L) && L > 0) {
                psi1 = -5 * (0.1 / L);
                psi2 = -5 * (2 / L);
                psi200 = -5 * (2 / L);
            } else {
                psi1 = 2 * logf((1 + x1 * x1) / 2);
                psi2 = 2 * logf((1 + x2 * x2) / 2);
                psi200 = 2 * logf((1 + x200) / 2) + logf((1 + x200 * x200) / 2) - 2 * atan(x200) + 0.5 * M_PI;
            }

            float ust = (VON_KARMAN * u200) / (logf(200 / products.zom[i]) - psi200);
            float rah = (logf(2 / 0.1) - psi2 + psi1) / (products.ustar[i] * VON_KARMAN);

            products.ustar[i] = ust;
            products.aerodynamic_resistance[i] = rah;
            products.sensible_heat_flux[i] = sensibleHeatFlux;
        }

        if ((i > 0) && (fabsf(1 - (rah_ini_pq_terra / products.aerodynamic_resistance[hot_pos])) < 0.05))
            break;
        else
            i++;
    }
    end = system_clock::now();

    general_time = duration_cast<nanoseconds>(end - begin).count() / 1000000.0;
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
    return "SERIAL,RAH_CYCLE," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
}

string sensible_heat_flux_function(Products products)
{
    int64_t initial_time, final_time;
    system_clock::time_point begin, end;
    float general_time;

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    begin = system_clock::now();
    for (int i = 0; i < products.height_band * products.width_band; i++) {
        products.sensible_heat_flux[i] = RHO * SPECIFIC_HEAT_AIR * (global_a + global_b * (products.surface_temperature[i])) / products.aerodynamic_resistance[i];
        if (!isnan(products.sensible_heat_flux[i]) && products.sensible_heat_flux[i] > (products.net_radiation[i] - products.soil_heat[i]))
            products.sensible_heat_flux[i] = products.net_radiation[i] - products.soil_heat[i];
    }
    end = system_clock::now();

    general_time = duration_cast<nanoseconds>(end - begin).count() / 1000000.0;
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
    return "SERIAL,SENSIBLE_HEAT_FLUX," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
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
    float u10 = (ustar_station / VON_KARMAN) * log(10 / station.SURFACE_ROUGHNESS);
    float u200 = (ustar_station / VON_KARMAN) * log(200 / station.SURFACE_ROUGHNESS);

    float ndvi_min = 1.0;
    float ndvi_max = -1.0;
    for (int i = 0; i < this->height_band * this->width_band; i++) {
        if (products.ndvi[i] < ndvi_min)
            ndvi_min = products.ndvi[i];
        if (products.ndvi[i] > ndvi_max)
            ndvi_max = products.ndvi[i];
    }

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
        result += rah_correction_function_blocks_ASEBAL(products, ndvi_min, ndvi_max, u200);
        result += sensible_heat_flux_function(products);
    }
    end = system_clock::now();

    general_time = duration_cast<nanoseconds>(end - begin).count() / 1000000.0;
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
    result += "SERIAL,P3_RAH," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
    return result;
};
