#include "kernels.cuh"

__device__ double a_d;
__device__ double b_d;

__global__ void d0_kernel(double *pai_d, double *d0_d, double CD1, double HGHT)
{
    unsigned int pos = threadIdx.x + blockIdx.x * blockDim.x;

    if (pos < width_d * height_d) {
        double cd1_pai_root = sqrtf(CD1 * pai_d[pos]);

        d0_d[pos] = HGHT * ((1.0 - (1.0 / cd1_pai_root)) + (pow(exp(1.0), -cd1_pai_root) / cd1_pai_root));
    }
}

__global__ void ustar_kernel_STEEP(double *zom_d, double *d0_d, double *ustar_d, double u10)
{
    unsigned int pos = threadIdx.x + blockIdx.x * blockDim.x;

    if (pos < width_d * height_d) {
        ustar_d[pos] = (u10 * VON_KARMAN) / logf((10 - d0_d[pos]) / zom_d[pos]);
    }
}

__global__ void ustar_kernel_ASEBAL(double *zom_d, double *ustar_d, double u200)
{
    unsigned int pos = threadIdx.x + blockIdx.x * blockDim.x;

    if (pos < width_d * height_d) {
        ustar_d[pos] = (u200 * VON_KARMAN) / logf(200 / zom_d[pos]);
    }
}

__global__ void zom_kernel_STEEP(double *d0_d, double *pai_d, double *zom_d, double A_ZOM, double B_ZOM)
{
    unsigned int pos = threadIdx.x + blockIdx.x * blockDim.x;

    double HGHT = 4;
    double CD = 0.01;
    double CR = 0.35;
    double PSICORR = 0.2;

    if (pos < width_d * height_d) {
        double gama = pow((CD + CR * (pai_d[pos] / 2)), -0.5);
        if (gama < 3.3)
            gama = 3.3;

        zom_d[pos] = (HGHT - d0_d[pos]) * exp(-VON_KARMAN * gama) + PSICORR;
    }
}

__global__ void zom_kernel_ASEBAL(double *ndvi_d, double *albedo_d, double *zom_d, double A_ZOM, double B_ZOM)
{
    unsigned int pos = threadIdx.x + blockIdx.x * blockDim.x;

    if (pos < width_d * height_d) {
        zom_d[pos] = exp((A_ZOM * ndvi_d[pos] / albedo_d[pos]) + B_ZOM);
    }
}

__global__ void kb_kernel(double *zom_d, double *ustar_d, double *pai_d, double *kb1_d, double *ndvi_d, double ndvi_max, double ndvi_min)
{
    unsigned int pos = threadIdx.x + blockIdx.x * blockDim.x;

    double HGHT = 4;

    double VON_KARMAN = 0.41;
    double visc = 0.00001461;
    double pr = 0.71;
    double c1 = 0.320;
    double c2 = 0.264;
    double c3 = 15.1;
    double cd = 0.2;
    double ct = 0.01;
    double sf_c = 0.3;
    double sf_d = 2.5;
    double sf_e = 4.0;
    double soil_moisture_day_rel = 0.33;

    if (pos < width_d * height_d) {
        double fc = 1 - pow((ndvi_d[pos] - ndvi_max) / (ndvi_min - ndvi_max), 0.4631);
        double fs = 1 - fc;

        double Re = (ustar_d[pos] * 0.009) / visc;
        double Ct = pow(pr, -0.667) * pow(Re, -0.5);
        double ratio = c1 - c2 * (exp(cd * -c3 * pai_d[pos]));
        double nec = (cd * pai_d[pos]) / (ratio * ratio * 2);
        double kbs = 2.46 * pow(Re, 0.25) - 2;

        double kb1_fst_part = (cd * VON_KARMAN) / (4 * ct * ratio * (1 - exp(nec * -0.5)));
        double kb1_sec_part = pow(fc, 2) + (VON_KARMAN * ratio * (zom_d[pos] / HGHT) / Ct);
        double kb1_trd_part = pow(fc, 2) * pow(fs, 2) + kbs * pow(fs, 2);
        double kb_ini = kb1_fst_part * kb1_sec_part * kb1_trd_part;

        double SF = sf_c + (1 / (1 + exp(sf_d - sf_e * soil_moisture_day_rel)));

        kb1_d[pos] = kb_ini * SF;
    }
}

__global__ void aerodynamic_resistance_kernel_STEEP(double *zom_d, double *d0_d, double *ustar_d, double *kb1_d, double *rah_d)
{
    unsigned int pos = threadIdx.x + blockIdx.x * blockDim.x;

    if (pos < width_d * height_d) {
        double rah_fst_part = 1 / (ustar_d[pos] * VON_KARMAN);
        double rah_sec_part = logf((10 - d0_d[pos]) / zom_d[pos]);
        double rah_trd_part = rah_fst_part * kb1_d[pos];
        rah_d[pos] = (rah_fst_part * rah_sec_part) + rah_trd_part;
    }
}

__global__ void aerodynamic_resistance_kernel_ASEBAL(double *ustar_d, double *rah_d)
{
    unsigned int pos = threadIdx.x + blockIdx.x * blockDim.x;

    if (pos < width_d * height_d) {
        rah_d[pos] = logf(2.0 / 0.1) / (ustar_d[pos] * VON_KARMAN);
    }
}

__global__ void rah_correction_cycle_STEEP(double *net_radiation_d, double *soil_heat_flux_d, double *ndvi_d, double *surface_temperature_d, double *d0_d, double *kb1_d, double *zom_d, double *ustar_d, double *rah_d, double *H_d, double ndvi_max, double ndvi_min)
{
    unsigned int pos = threadIdx.x + blockIdx.x * blockDim.x;

    if (pos < width_d * height_d) {
        unsigned int hot_pos = hotEndmemberLine_d * width_d + hotEndmemberCol_d;
        unsigned int cold_pos = coldEndmemberLine_d * width_d + coldEndmemberCol_d;

        double rah_ini_hot = rah_d[hot_pos];
        double rah_ini_cold = rah_d[cold_pos];

        double fc_hot = 1.0 - pow((ndvi_d[hot_pos] - ndvi_max) / (ndvi_min - ndvi_max), 0.4631);
        double fc_cold = 1.0 - pow((ndvi_d[cold_pos] - ndvi_max) / (ndvi_min - ndvi_max), 0.4631);

        double LE_hot = 0.55 * fc_hot * (net_radiation_d[hot_pos] - soil_heat_flux_d[hot_pos]) * 0.78;
        double LE_cold = 1.75 * fc_cold * (net_radiation_d[cold_pos] - soil_heat_flux_d[cold_pos]) * 0.78;

        double H_cold = net_radiation_d[cold_pos] - soil_heat_flux_d[cold_pos] - LE_hot;
        double dt_cold = H_cold * rah_ini_cold / (RHO * SPECIFIC_HEAT_AIR);

        double H_hot = net_radiation_d[hot_pos] - soil_heat_flux_d[hot_pos] - LE_cold;
        double dt_hot = H_hot * rah_ini_hot / (RHO * SPECIFIC_HEAT_AIR);

        double b = (dt_hot - dt_cold) / (surface_temperature_d[hot_pos] - surface_temperature_d[cold_pos]);
        double a = dt_cold - (b * surface_temperature_d[cold_pos]);

        b_d = b;
        a_d = a;

        double dt_final = a + b * surface_temperature_d[pos];
        H_d[pos] = RHO * SPECIFIC_HEAT_AIR * dt_final / rah_d[pos];
        double L = -1.0 * ((RHO * SPECIFIC_HEAT_AIR * pow(ustar_d[pos], 3.0) * surface_temperature_d[pos]) / (VON_KARMAN * GRAVITY * H_d[pos]));

        double y2 = pow((1.0 - (16.0 * (10.0 - d0_d[pos])) / L), 0.25);
        double x200 = pow((1.0 - (16.0 * (10.0 - d0_d[pos])) / L), 0.25);

        double psi2, psi200;
        if (!isnan(L) && L > 0) {
            psi2 = -5.0 * ((10.0 - d0_d[pos]) / L);
            psi200 = -5.0 * ((10.0 - d0_d[pos]) / L);
        } else {
            psi2 = 2.0 * logf((1.0 + y2 * y2) / 2.0);
            psi200 = 2.0 * logf((1.0 + x200) / 2.0) + logf((1.0 + x200 * x200) / 2.0) - 2.0 * atanf(x200) + 0.5 * M_PI;
        }

        ustar_d[pos] = (VON_KARMAN * ustar_d[pos]) / (logf((10.0 - d0_d[pos]) / zom_d[pos]) - psi200);

        double rah_fst_part = 1.0 / (ustar_d[pos] * VON_KARMAN);
        double rah_sec_part = logf((10.0 - d0_d[pos]) / zom_d[pos]) - psi2;
        double rah_trd_part = rah_fst_part * kb1_d[pos];
        rah_d[pos] = (rah_fst_part * rah_sec_part) + rah_trd_part;
    }
}

__global__ void rah_correction_cycle_ASEBAL(double *net_radiation_d, double *soil_heat_flux_d, double *ndvi_d, double *surface_temperature_d, double *kb1_d, double *zom_d, double *ustar_d, double *rah_d, double *H_d, double u200, int *stop_condition)
{
    unsigned int pos = threadIdx.x + blockIdx.x * blockDim.x;

    if (pos < width_d * height_d) {
        unsigned int hot_pos = hotEndmemberLine_d * width_d + hotEndmemberCol_d;
        unsigned int cold_pos = coldEndmemberLine_d * width_d + coldEndmemberCol_d;

        double rah_ini_hot = rah_d[hot_pos];
        double rah_ini_cold = rah_d[cold_pos];

        double H_cold = net_radiation_d[cold_pos] - soil_heat_flux_d[cold_pos];
        double dt_cold = H_cold * rah_ini_cold / (RHO * SPECIFIC_HEAT_AIR);

        double H_hot = net_radiation_d[hot_pos] - soil_heat_flux_d[hot_pos];
        double dt_hot = H_hot * rah_ini_hot / (RHO * SPECIFIC_HEAT_AIR);

        double b = (dt_hot - dt_cold) / (surface_temperature_d[hot_pos] - surface_temperature_d[cold_pos]);
        double a = dt_cold - (b * surface_temperature_d[cold_pos]);

        b_d = b;
        a_d = a;

        double dt_final = a + b * (surface_temperature_d[pos]);

        H_d[pos] = RHO * SPECIFIC_HEAT_AIR * (dt_final) / rah_d[pos];
        double L = -1.0 * ((RHO * SPECIFIC_HEAT_AIR * pow(ustar_d[pos], 3.0) * surface_temperature_d[pos]) / (VON_KARMAN * GRAVITY * H_d[pos]));

        double x1 = pow((1.0 - (16.0 * 0.1) / L), 0.25);
        double x2 = pow((1.0 - (16.0 * 2.0) / L), 0.25);
        double x200 = pow((1.0 - (16.0 * 200.0) / L), 0.25);

        double psi1, psi2, psi200;
        if (!isnan(L) && L > 0) {
            psi1 = -5.0 * (0.1 / L);
            psi2 = -5.0 * (2.0 / L);
            psi200 = -5.0 * (2.0 / L);
        } else {
            psi1 = 2.0 * logf((1.0 + x1 * x1) / 2.0);
            psi2 = 2.0 * logf((1.0 + x2 * x2) / 2.0);
            psi200 = 2.0 * logf((1.0 + x200) / 2.0) + logf((1.0 + x200 * x200) / 2.0) - 2.0 * atanf(x200) + 0.5 * M_PI;
        }

        ustar_d[pos] = (VON_KARMAN * u200) / (logf(200.0 / zom_d[pos]) - psi200);
        rah_d[pos] = (logf(2.0 / 0.1) - psi2 + psi1) / (ustar_d[pos] * VON_KARMAN);

        if ((pos == hot_pos) && (fabsf(1.0 - (rah_ini_hot / rah_d[hot_pos])) < 0.05)) {
            atomicExch(stop_condition, 1);
        }
    }
}

__global__ void sensible_heat_flux_kernel(double *surface_temperature_d, double *rah_d, double *net_radiation_d, double *soil_heat_d, double *sensible_heat_flux_d)
{
    unsigned int pos = threadIdx.x + blockIdx.x * blockDim.x;

    if (pos < width_d * height_d) {
        sensible_heat_flux_d[pos] = RHO * SPECIFIC_HEAT_AIR * (a_d + b_d * surface_temperature_d[pos]) / rah_d[pos];
        if (!isnan(sensible_heat_flux_d[pos]) && sensible_heat_flux_d[pos] > (net_radiation_d[pos] - soil_heat_d[pos]))
            sensible_heat_flux_d[pos] = net_radiation_d[pos] - soil_heat_d[pos];
    }
}
