#include "kernels.cuh"

__device__ float a_d;
__device__ float b_d;

__global__ void d0_kernel(float *pai_d, float *d0_d, float CD1, float HGHT)
{
    unsigned int pos = threadIdx.x + blockIdx.x * blockDim.x;

    if (pos < width_d * height_d) {
        float cd1_pai_root = sqrt(CD1 * pai_d[pos]);

        d0_d[pos] = HGHT * ((1 - (1 / cd1_pai_root)) + (pow(exp(1.0), -cd1_pai_root) / cd1_pai_root));
    }
}

__global__ void ustar_kernel_STEEP(float *zom_d, float *d0_d, float *ustar_d, float u10)
{
    unsigned int pos = threadIdx.x + blockIdx.x * blockDim.x;

    if (pos < width_d * height_d) {
        ustar_d[pos] = (u10 * VON_KARMAN) / logf((10 - d0_d[pos]) / zom_d[pos]);
    }
}

__global__ void ustar_kernel_ASEBAL(float *zom_d, float *ustar_d, float u200)
{
    unsigned int pos = threadIdx.x + blockIdx.x * blockDim.x;

    if (pos < width_d * height_d) {
        ustar_d[pos] = (u200 * VON_KARMAN) / logf(200 / zom_d[pos]);
    }
}

__global__ void zom_kernel_STEEP(float *d0_d, float *pai_d, float *zom_d, float A_ZOM, float B_ZOM)
{
    unsigned int pos = threadIdx.x + blockIdx.x * blockDim.x;

    float HGHT = 4;
    float CD = 0.01;
    float CR = 0.35;
    float PSICORR = 0.2;

    if (pos < width_d * height_d) {
        float gama = pow((CD + CR * (pai_d[pos] / 2)), -0.5);
        if (gama < 3.3)
            gama = 3.3;

        zom_d[pos] = (HGHT - d0_d[pos]) * exp(-VON_KARMAN * gama) + PSICORR;
    }
}

__global__ void zom_kernel_ASEBAL(float *ndvi_d, float *albedo_d, float *zom_d, float A_ZOM, float B_ZOM)
{
    unsigned int pos = threadIdx.x + blockIdx.x * blockDim.x;

    if (pos < width_d * height_d) {
        zom_d[pos] = exp((A_ZOM * ndvi_d[pos] / albedo_d[pos]) + B_ZOM);
    }
}

__global__ void kb_kernel(float *zom_d, float *ustar_d, float *pai_d, float *kb1_d, float *ndvi_d, float ndvi_max, float ndvi_min)
{
    unsigned int pos = threadIdx.x + blockIdx.x * blockDim.x;

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

    if (pos < width_d * height_d) {
        float fc = 1 - pow((ndvi_d[pos] - ndvi_max) / (ndvi_min - ndvi_max), 0.4631);
        float fs = 1 - fc;

        float Re = (ustar_d[pos] * 0.009) / visc;
        float Ct = pow(pr, -0.667) * pow(Re, -0.5);
        float ratio = c1 - c2 * (exp(cd * -c3 * pai_d[pos]));
        float nec = (cd * pai_d[pos]) / (ratio * ratio * 2);
        float kbs = 2.46 * pow(Re, 0.25) - 2;

        float kb1_fst_part = (cd * VON_KARMAN) / (4 * ct * ratio * (1 - exp(nec * -0.5)));
        float kb1_sec_part = pow(fc, 2) + (VON_KARMAN * ratio * (zom_d[pos] / HGHT) / Ct);
        float kb1_trd_part = pow(fc, 2) * pow(fs, 2) + kbs * pow(fs, 2);
        float kb_ini = kb1_fst_part * kb1_sec_part * kb1_trd_part;

        float SF = sf_c + (1 / (1 + exp(sf_d - sf_e * soil_moisture_day_rel)));

        kb1_d[pos] = kb_ini * SF;
    }
}

__global__ void aerodynamic_resistance_kernel_STEEP(float *zom_d, float *d0_d, float *ustar_d, float *kb1_d, float *rah_d)
{
    unsigned int pos = threadIdx.x + blockIdx.x * blockDim.x;

    if (pos < width_d * height_d) {
        float rah_fst_part = 1 / (ustar_d[pos] * VON_KARMAN);
        float rah_sec_part = logf((10 - d0_d[pos]) / zom_d[pos]);
        float rah_trd_part = rah_fst_part * kb1_d[pos];
        rah_d[pos] = (rah_fst_part * rah_sec_part) + rah_trd_part;
    }
}

__global__ void aerodynamic_resistance_kernel_ASEBAL(float *ustar_d, float *rah_d)
{
    unsigned int pos = threadIdx.x + blockIdx.x * blockDim.x;

    if (pos < width_d * height_d) {
        rah_d[pos] = logf(2.0 / 0.1) / (ustar_d[pos] * VON_KARMAN);
    }
}

__global__ void rah_correction_cycle_STEEP(float *net_radiation_d, float *soil_heat_flux_d, float *ndvi_d, float *surface_temperature_d, float *d0_d, float *kb1_d, float *zom_d, float *ustar_d, float *rah_d, float *H_d, float ndvi_max, float ndvi_min)
{
    unsigned int pos = threadIdx.x + blockIdx.x * blockDim.x;

    if (pos < width_d * height_d) {
        unsigned int hot_pos = hotEndmemberLine_d * width_d + hotEndmemberCol_d;
        unsigned int cold_pos = coldEndmemberLine_d * width_d + coldEndmemberCol_d;

        float rah_ini_hot = rah_d[hot_pos];
        float rah_ini_cold = rah_d[cold_pos];

        float fc_hot = 1 - pow((ndvi_d[hot_pos] - ndvi_max) / (ndvi_min - ndvi_max), 0.4631);
        float fc_cold = 1 - pow((ndvi_d[cold_pos] - ndvi_max) / (ndvi_min - ndvi_max), 0.4631);

        float LE_hot = 0.55 * fc_hot * (net_radiation_d[hot_pos] - soil_heat_flux_d[hot_pos]) * 0.78;
        float LE_cold = 1.75 * fc_cold * (net_radiation_d[cold_pos] - soil_heat_flux_d[cold_pos]) * 0.78;

        float H_cold = net_radiation_d[cold_pos] - soil_heat_flux_d[cold_pos] - LE_hot;
        float dt_cold = H_cold * rah_ini_cold / (RHO * SPECIFIC_HEAT_AIR);

        float H_hot = net_radiation_d[hot_pos] - soil_heat_flux_d[hot_pos] - LE_cold;
        float dt_hot = H_hot * rah_ini_hot / (RHO * SPECIFIC_HEAT_AIR);

        float b = (dt_hot - dt_cold) / (surface_temperature_d[hot_pos] - surface_temperature_d[cold_pos]);
        float a = dt_cold - (b * surface_temperature_d[cold_pos]);

        b_d = b;
        a_d = a;

        float dt_final = a + b * surface_temperature_d[pos];
        H_d[pos] = RHO * SPECIFIC_HEAT_AIR * dt_final / rah_d[pos];
        float L = -1 * ((RHO * SPECIFIC_HEAT_AIR * pow(ustar_d[pos], 3) * surface_temperature_d[pos]) / (VON_KARMAN * GRAVITY * H_d[pos]));

        float y2 = pow((1 - (16 * (10 - d0_d[pos])) / L), 0.25);
        float x200 = pow((1 - (16 * (10 - d0_d[pos])) / L), 0.25);

        float psi2, psi200;
        if (!isnan(L) && L > 0) {
            psi2 = -5 * ((10 - d0_d[pos]) / L);
            psi200 = -5 * ((10 - d0_d[pos]) / L);
        } else {
            psi2 = 2 * logf((1 + y2 * y2) / 2);
            psi200 = 2 * logf((1 + x200) / 2) + logf((1 + x200 * x200) / 2) - 2 * atan(x200) + 0.5 * M_PI;
        }

        ustar_d[pos] = (VON_KARMAN * ustar_d[pos]) / (logf((10 - d0_d[pos]) / zom_d[pos]) - psi200);

        float rah_fst_part = 1 / (ustar_d[pos] * VON_KARMAN);
        float rah_sec_part = logf((10 - d0_d[pos]) / zom_d[pos]) - psi2;
        float rah_trd_part = rah_fst_part * kb1_d[pos];
        rah_d[pos] = (rah_fst_part * rah_sec_part) + rah_trd_part;
    }
}

__global__ void rah_correction_cycle_ASEBAL(float *net_radiation_d, float *soil_heat_flux_d, float *ndvi_d, float *surface_temperature_d, float *kb1_d, float *zom_d, float *ustar_d, float *rah_d, float *H_d, float u200, int *stop_condition)
{
    unsigned int pos = threadIdx.x + blockIdx.x * blockDim.x;

    if (pos < width_d * height_d) {
        unsigned int hot_pos = hotEndmemberLine_d * width_d + hotEndmemberCol_d;
        unsigned int cold_pos = coldEndmemberLine_d * width_d + coldEndmemberCol_d;

        float rah_ini_hot = rah_d[hot_pos];
        float rah_ini_cold = rah_d[cold_pos];

        float H_cold = net_radiation_d[cold_pos] - soil_heat_flux_d[cold_pos];
        float dt_cold = H_cold * rah_ini_cold / (RHO * SPECIFIC_HEAT_AIR);

        float H_hot = net_radiation_d[hot_pos] - soil_heat_flux_d[hot_pos];
        float dt_hot = H_hot * rah_ini_hot / (RHO * SPECIFIC_HEAT_AIR);

        float b = (dt_hot - dt_cold) / (surface_temperature_d[hot_pos] - surface_temperature_d[cold_pos]);
        float a = dt_cold - (b * surface_temperature_d[cold_pos]);

        b_d = b;
        a_d = a;

        float dt_final = a + b * (surface_temperature_d[pos]);

        H_d[pos] = RHO * SPECIFIC_HEAT_AIR * (dt_final) / rah_d[pos];
        float L = -1 * ((RHO * SPECIFIC_HEAT_AIR * pow(ustar_d[pos], 3) * surface_temperature_d[pos]) / (VON_KARMAN * GRAVITY * H_d[pos]));

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

        ustar_d[pos] = (VON_KARMAN * u200) / (logf(200 / zom_d[pos]) - psi200);
        rah_d[pos] = (logf(2 / 0.1) - psi2 + psi1) / (ustar_d[pos] * VON_KARMAN);

        if ((pos == hot_pos) && (fabsf(1 - (rah_ini_hot / rah_d[hot_pos])) < 0.05)) {
            atomicExch(stop_condition, 1);
        }
    }
}

__global__ void sensible_heat_flux_kernel(float *surface_temperature_d, float *rah_d, float *net_radiation_d, float *soil_heat_d, float *sensible_heat_flux_d)
{
    unsigned int pos = threadIdx.x + blockIdx.x * blockDim.x;

    if (pos < width_d * height_d) {
        sensible_heat_flux_d[pos] = RHO * SPECIFIC_HEAT_AIR * (a_d + b_d * surface_temperature_d[pos]) / rah_d[pos];
        if (!isnan(sensible_heat_flux_d[pos]) && sensible_heat_flux_d[pos] > (net_radiation_d[pos] - soil_heat_d[pos]))
            sensible_heat_flux_d[pos] = net_radiation_d[pos] - soil_heat_d[pos];
    }
}
