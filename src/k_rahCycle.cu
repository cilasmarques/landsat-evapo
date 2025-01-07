#include "kernels.cuh"

__shared__ float a_d;
__shared__ float b_d;

__global__ void ustar_kernel_STEEP(float *zom_d, float *d0_d, float *ustar_d, float u10)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < width_d * height_d) {
        unsigned int row = idx / width_d;
        unsigned int col = idx % width_d;
        unsigned int pos = row * width_d + col;
        ustar_d[pos] = (u10 * VON_KARMAN) / logf((10 - d0_d[pos]) / zom_d[pos]);
    }
}

__global__ void ustar_kernel_ASEBAL(float *zom_d, float *ustar_d, float u200)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < width_d * height_d) {
        unsigned int row = idx / width_d;
        unsigned int col = idx % width_d;
        unsigned int pos = row * width_d + col;
        ustar_d[pos] = (u200 * VON_KARMAN) / logf(200 / zom_d[pos]);
    }
}

__global__ void rah_correction_cycle_STEEP(float *net_radiation_d, float *soil_heat_flux_d, float *ndvi_d, float *surface_temperature_d, float *d0_d, float *kb1_d, float *zom_d, float *ustar_d, float *rah_d, float *H_d, float ndvi_max, float ndvi_min)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < width_d * height_d) {
        unsigned int row = idx / width_d;
        unsigned int col = idx % width_d;
        unsigned int pos = row * width_d + col;

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

        float dt_final = a + (b * surface_temperature_d[pos]);

        float sensibleHeatFlux = RHO * SPECIFIC_HEAT_AIR * (dt_final) / rah_d[pos];
        float L = -1 * ((RHO * SPECIFIC_HEAT_AIR * pow(ustar_d[pos], 3) * surface_temperature_d[pos]) / (VON_KARMAN * GRAVITY * sensibleHeatFlux));

        float DISP = d0_d[pos];
        float y2 = pow((1 - (16 * (10 - d0_d[pos])) / L), 0.25);
        float x200 = pow((1 - (16 * (10 - DISP)) / L), 0.25);

        float psi2, psi200;
        if (!isnan(L) && L > 0) {
            psi2 = -5 * ((10 - DISP) / L);
            psi200 = -5 * ((10 - DISP) / L);
        } else {
            psi2 = 2 * logf((1 + y2 * y2) / 2);
            psi200 = 2 * logf((1 + x200) / 2) + logf((1 + x200 * x200) / 2) - 2 * atan(x200) + 0.5 * M_PI;
        }

        float ust = (VON_KARMAN * ustar_d[pos]) / (logf((10 - DISP) / zom_d[pos]) - psi200);

        float rah_fst_part = 1 / (ustar_d[pos] * VON_KARMAN);
        float rah_sec_part = logf((10 - d0_d[pos]) / zom_d[pos]) - psi2;
        float rah_trd_part = rah_fst_part * kb1_d[pos];
        float rah = (rah_fst_part * rah_sec_part) + rah_trd_part;

        ustar_d[pos] = ust;
        rah_d[pos] = rah;
        H_d[pos] = sensibleHeatFlux;
    }
}

__global__ void rah_correction_cycle_ASEBAL(float *net_radiation_d, float *soil_heat_flux_d, float *ndvi_d, float *surface_temperature_d, float *kb1_d, float *zom_d, float *ustar_d, float *rah_d, float *H_d, float u200, int *stop_condition)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < width_d * height_d) {
        unsigned int row = idx / width_d;
        unsigned int col = idx % width_d;
        unsigned int pos = row * width_d + col;

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

        float sensibleHeatFlux = RHO * SPECIFIC_HEAT_AIR * (dt_final) / rah_d[pos];
        float L = -1 * ((RHO * SPECIFIC_HEAT_AIR * pow(ustar_d[pos], 3) * surface_temperature_d[pos]) / (VON_KARMAN * GRAVITY * sensibleHeatFlux));

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

        float ust = (VON_KARMAN * u200) / (logf(200 / zom_d[pos]) - psi200);
        float rah = (logf(2 / 0.1) - psi2 + psi1) / (ustar_d[pos] * VON_KARMAN);

        if ((pos == hot_pos) && (fabsf(1 - (rah_ini_hot / rah)) < 0.05)) {
            atomicExch(stop_condition, 1);
        }

        ustar_d[pos] = ust;
        rah_d[pos] = rah;
        H_d[pos] = sensibleHeatFlux;
    }
}

__global__ void sensible_heat_flux_kernel(float *surface_temperature_d, float *rah_d, float *net_radiation_d, float *soil_heat_d, float *sensible_heat_flux_d)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < width_d * height_d) {
        unsigned int row = idx / width_d;
        unsigned int col = idx % width_d;
        unsigned int pos = row * width_d + col;

        sensible_heat_flux_d[pos] = RHO * SPECIFIC_HEAT_AIR * (a_d + b_d * (surface_temperature_d[pos])) / rah_d[pos];
        if (!isnan(sensible_heat_flux_d[pos]) && sensible_heat_flux_d[pos] > (net_radiation_d[pos] - soil_heat_d[pos]))
            sensible_heat_flux_d[pos] = net_radiation_d[pos] - soil_heat_d[pos];
    }
}
