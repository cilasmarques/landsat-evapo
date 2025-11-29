#include "constants.h"
#include "kernels.cuh"

__device__ float a_d;
__device__ float b_d;

__global__ void d0_kernel(float *pai_d, float *d0_d, float CD1, float HGHT)
{
    unsigned int pos = threadIdx.x + blockIdx.x * blockDim.x;

    if (pos < width_d * height_d) {
        float cd1_pai_root = sqrtf(CD1 * pai_d[pos]);

        d0_d[pos] = HGHT * ((1.0f - (1.0f / cd1_pai_root)) + (__expf(-cd1_pai_root) / cd1_pai_root));
    }
}

__global__ void ustar_kernel_STEEP(float *zom_d, float *d0_d, float *ustar_d, float u10)
{
    unsigned int pos = threadIdx.x + blockIdx.x * blockDim.x;

    if (pos < width_d * height_d) {
        ustar_d[pos] = (u10 * VON_KARMAN) / __logf((10.0f - d0_d[pos]) / zom_d[pos]);
    }
}

__global__ void ustar_kernel_ASEBAL(float *zom_d, float *ustar_d, float u200)
{
    unsigned int pos = threadIdx.x + blockIdx.x * blockDim.x;

    if (pos < width_d * height_d) {
        ustar_d[pos] = (u200 * VON_KARMAN) / __logf(200.0f / zom_d[pos]);
    }
}

__global__ void zom_kernel_STEEP(float *d0_d, float *pai_d, float *zom_d, float A_ZOM, float B_ZOM)
{
    unsigned int pos = threadIdx.x + blockIdx.x * blockDim.x;

    float HGHT = 4.0f;
    float CD = 0.01f;
    float CR = 0.35f;
    float PSICORR = 0.2f;

    if (pos < width_d * height_d) {
        float gama = __powf((CD + CR * (pai_d[pos] / 2.0f)), -0.5f);
        if (gama < 3.3f)
            gama = 3.3f;

        zom_d[pos] = (HGHT - d0_d[pos]) * __expf(-VON_KARMAN * gama) + PSICORR;
    }
}

__global__ void zom_kernel_ASEBAL(float *ndvi_d, float *albedo_d, float *zom_d, float A_ZOM, float B_ZOM)
{
    unsigned int pos = threadIdx.x + blockIdx.x * blockDim.x;

    if (pos < width_d * height_d) {
        zom_d[pos] = __expf((A_ZOM * ndvi_d[pos] / albedo_d[pos]) + B_ZOM);
    }
}

__global__ void kb_kernel(float *zom_d, float *ustar_d, float *pai_d, float *kb1_d, float *ndvi_d, float ndvi_max, float ndvi_min)
{
    unsigned int pos = threadIdx.x + blockIdx.x * blockDim.x;

    float HGHT = 4.0f;

    float VON_KARMAN = 0.41f;
    float visc = 0.00001461f;
    float pr = 0.71f;
    float c1 = 0.320f;
    float c2 = 0.264f;
    float c3 = 15.1f;
    float cd = 0.2f;
    float ct = 0.01f;
    float sf_c = 0.3f;
    float sf_d = 2.5f;
    float sf_e = 4.0f;
    float soil_moisture_day_rel = 0.33f;

    if (pos < width_d * height_d) {
        float fc = 1.0f - __powf((ndvi_d[pos] - ndvi_max) / (ndvi_min - ndvi_max), 0.4631f);
        float fs = 1.0f - fc;

        float Re = (ustar_d[pos] * 0.009f) / visc;
        float Ct = __powf(pr, -0.667f) * __powf(Re, -0.5f);
        float ratio = c1 - c2 * (__expf(cd * -c3 * pai_d[pos]));
        float nec = (cd * pai_d[pos]) / (ratio * ratio * 2.0f);
        float kbs = 2.46f * __powf(Re, 0.25f) - 2.0f;

        float kb1_fst_part = (cd * VON_KARMAN) / (4.0f * ct * ratio * (1.0f - __expf(nec * -0.5f)));
        float kb1_sec_part = __powf(fc, 2.0f) + (VON_KARMAN * ratio * (zom_d[pos] / HGHT) / Ct);
        float kb1_trd_part = __powf(fc, 2.0f) * __powf(fs, 2.0f) + kbs * __powf(fs, 2.0f);
        float kb_ini = kb1_fst_part * kb1_sec_part * kb1_trd_part;

	float SF = sf_c + (1.0f / (1.0f + __expf(-sf_d) - sf_e * soil_moisture_day_rel));

        kb1_d[pos] = kb_ini * SF;
    }
}

__global__ void aerodynamic_resistance_kernel_STEEP(float *zom_d, float *d0_d, float *ustar_d, float *kb1_d, float *rah_d)
{
    unsigned int pos = threadIdx.x + blockIdx.x * blockDim.x;

    if (pos < width_d * height_d) {
        float rah_fst_part = 1.0f / (ustar_d[pos] * VON_KARMAN);
        float rah_sec_part = __logf((10.0f - d0_d[pos]) / zom_d[pos]);
        float rah_trd_part = rah_fst_part * kb1_d[pos];
        rah_d[pos] = (rah_fst_part * rah_sec_part) + rah_trd_part;
    }
}

__global__ void aerodynamic_resistance_kernel_ASEBAL(float *ustar_d, float *rah_d)
{
    unsigned int pos = threadIdx.x + blockIdx.x * blockDim.x;

    if (pos < width_d * height_d) {
        rah_d[pos] = __logf(2.0f / 0.1f) / (ustar_d[pos] * VON_KARMAN);
    }
}

__global__ void rah_correction_cycle_STEEP(float *surface_temperature_d, float *d0_d, float *kb1_d, float *zom_d, float *ustar_d, float *rah_d, float *H_d)
{
    unsigned int pos = threadIdx.x + blockIdx.x * blockDim.x;

    if (pos < width_d * height_d) {
        float dt_final = a_d + b_d * surface_temperature_d[pos];
        H_d[pos] = RHO * SPECIFIC_HEAT_AIR * dt_final / rah_d[pos];
        float L = -1.0f * ((RHO * SPECIFIC_HEAT_AIR * __powf(ustar_d[pos], 3.0f) * surface_temperature_d[pos]) / (VON_KARMAN * GRAVITY * H_d[pos]));

        float y2 = __powf((1.0f - (16.0f * (10.0f - d0_d[pos])) / L), 0.25f);
        float x200 = __powf((1.0f - (16.0f * (10.0f - d0_d[pos])) / L), 0.25f);

        float psi2, psi200;
        if (!isnan(L) && L > 0) {
            psi2 = -5.0f * ((10.0f - d0_d[pos]) / L);
            psi200 = -5.0f * ((10.0f - d0_d[pos]) / L);
        } else {
            psi2 = 2.0f * __logf((1.0f + y2 * y2) / 2.0f);
            psi200 = 2.0f * __logf((1.0f + x200) / 2.0f) + __logf((1.0f + x200 * x200) / 2.0f) - 2.0f * atanf(x200) + 0.5f * PI;
        }

        ustar_d[pos] = (VON_KARMAN * ustar_d[pos]) / (__logf((10.0f - d0_d[pos]) / zom_d[pos]) - psi200);

        float rah_fst_part = 1.0f / (ustar_d[pos] * VON_KARMAN);
        float rah_sec_part = __logf((10.0f - d0_d[pos]) / zom_d[pos]) - psi2;
        float rah_trd_part = rah_fst_part * kb1_d[pos];
        rah_d[pos] = (rah_fst_part * rah_sec_part) + rah_trd_part;
    }
}

__global__ void rah_correction_cycle_ASEBAL(float *surface_temperature_d, float *zom_d, float *ustar_d, float *rah_d, float *H_d, float u200)
{
    unsigned int pos = threadIdx.x + blockIdx.x * blockDim.x;

    if (pos < width_d * height_d) {
        float dt_final = a_d + b_d * (surface_temperature_d[pos]);
        H_d[pos] = RHO * SPECIFIC_HEAT_AIR * (dt_final) / rah_d[pos];
        float L = -1.0f * ((RHO * SPECIFIC_HEAT_AIR * __powf(ustar_d[pos], 3.0f) * surface_temperature_d[pos]) / (VON_KARMAN * GRAVITY * H_d[pos]));

        float x1 = __powf((1.0f - (16.0f * 0.1f) / L), 0.25f);
        float x2 = __powf((1.0f - (16.0f * 2.0f) / L), 0.25f);
        float x200 = __powf((1.0f - (16.0f * 200.0f) / L), 0.25f);

        float psi1, psi2, psi200;
        if (!isnan(L) && L > 0) {
            psi1 = -5.0f * (0.1f / L);
            psi2 = -5.0f * (2.0f / L);
            psi200 = -5.0f * (2.0f / L);
        } else {
            psi1 = 2.0f * __logf((1.0f + x1 * x1) / 2.0f);
            psi2 = 2.0f * __logf((1.0f + x2 * x2) / 2.0f);
            psi200 = 2.0f * __logf((1.0f + x200) / 2.0f) + __logf((1.0f + x200 * x200) / 2.0f) - 2.0f * atanf(x200) + 0.5f * PI;
        }

        ustar_d[pos] = (VON_KARMAN * u200) / (__logf(200.0f / zom_d[pos]) - psi200);
        rah_d[pos] = (__logf(2.0f / 0.1) - psi2 + psi1) / (ustar_d[pos] * VON_KARMAN);
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
