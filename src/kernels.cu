#include "kernels.cuh"

__device__ int width_d;
__device__ int height_d;

__device__ int hotEndmemberLine_d;
__device__ int hotEndmemberCol_d;
__device__ int coldEndmemberLine_d;
__device__ int coldEndmemberCol_d;

__global__ void NAN_kernel(float *pointer_d)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < width_d * height_d) {
        unsigned int row = idx / width_d;
        unsigned int col = idx % width_d;
        unsigned int pos = row * width_d + col;

        if (pointer_d[pos] <= 0)
            pointer_d[pos] = NAN;
    }
}

__global__ void lai_kernel(float *reflectance_nir_d, float *reflectance_red_d, float *lai_d)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < width_d * height_d) {
        unsigned int row = idx / width_d;
        unsigned int col = idx % width_d;
        unsigned int pos = row * width_d + col;

        float savi = ((1 + 0.5) * (reflectance_nir_d[pos] - reflectance_red_d[pos])) / (0.5 + (reflectance_nir_d[pos] + reflectance_red_d[pos]));

        if (!isnan(savi) && savi > 0.687)
            lai_d[pos] = 6;
        if (!isnan(savi) && savi <= 0.687)
            lai_d[pos] = -logf((0.69 - savi) / 0.59) / 0.91;
        if (!isnan(savi) && savi < 0.1)
            lai_d[pos] = 0;

        if (lai_d[pos] < 0)
            lai_d[pos] = 0;
    }
}

__global__ void enb_kernel(float *lai_d, float *ndvi_d, float *enb_d)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < width_d * height_d) {
        unsigned int row = idx / width_d;
        unsigned int col = idx % width_d;
        unsigned int pos = row * width_d + col;

        if (ndvi_d[pos] > 0)
            enb_d[pos] = (lai_d[pos] < 3) ? 0.97 + 0.0033 * lai_d[pos] : 0.98;            
        else if (ndvi_d[pos] < 0)
            enb_d[pos] = 0.99;
        else
            enb_d[pos] = NAN;
    }
}

__global__ void eo_kernel(float *lai_d, float *ndvi_d, float *eo_d)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < width_d * height_d) {
        unsigned int row = idx / width_d;
        unsigned int col = idx % width_d;
        unsigned int pos = row * width_d + col;

        if (ndvi_d[pos] > 0)
            eo_d[pos] = (lai_d[pos] < 3) ? 0.95 + 0.01 * lai_d[pos] : 0.98;            
        else if (ndvi_d[pos] < 0)
            eo_d[pos] = 0.985;
        else
            eo_d[pos] = NAN;
    }
}

__global__ void surface_temperature_kernel(float *enb_d, float *radiance_termal_d, float *surface_temperature_d, float k1, float k2)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < width_d * height_d) {
        unsigned int row = idx / width_d;
        unsigned int col = idx % width_d;
        unsigned int pos = row * width_d + col;
        surface_temperature_d[pos] = k2 / (logf((enb_d[pos] * k1 / radiance_termal_d[pos]) + 1));

        if (surface_temperature_d[pos] < 0)
            surface_temperature_d[pos] = 0;
    }
}

__global__ void rah_correction_cycle_STEEP(float *net_radiation_d, float *soil_heat_flux_d, float *ndvi_d, float *surf_temp_d, float *d0_d, float *kb1_d, float *zom_d, float *ustar_d, float *rah_d, float *H_d, float *a_d, float *b_d, float ndvi_max, float ndvi_min)
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

        float b = (dt_hot - dt_cold) / (surf_temp_d[hot_pos] - surf_temp_d[cold_pos]);
        float a = dt_cold - (b * surf_temp_d[cold_pos]);

        *b_d = b;
        *a_d = a;

        float dt_final = a + (b * surf_temp_d[pos]);

        float sensibleHeatFlux = RHO * SPECIFIC_HEAT_AIR * (dt_final) / rah_d[pos];
        float L = -1 * ((RHO * SPECIFIC_HEAT_AIR * pow(ustar_d[pos], 3) * surf_temp_d[pos]) / (VON_KARMAN * GRAVITY * sensibleHeatFlux));

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

__global__ void rah_correction_cycle_ASEBAL(float *net_radiation_d, float *soil_heat_flux_d, float *ndvi_d, float *surf_temp_d, float *kb1_d, float *zom_d, float *ustar_d, float *rah_d, float *H_d, float *a_d, float *b_d, float u200, int *stop_condition)
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

        float b = (dt_hot - dt_cold) / (surf_temp_d[hot_pos] - surf_temp_d[cold_pos]);
        float a = dt_cold - (b * surf_temp_d[cold_pos]);

        *b_d = b;
        *a_d = a;

        float dt_final = a + b * (surf_temp_d[pos]);

        float sensibleHeatFlux = RHO * SPECIFIC_HEAT_AIR * (dt_final) / rah_d[pos];
        float L = -1 * ((RHO * SPECIFIC_HEAT_AIR * pow(ustar_d[pos], 3) * surf_temp_d[pos]) / (VON_KARMAN * GRAVITY * sensibleHeatFlux));

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

__global__ void filter_valid_values(const float *target, float *filtered, int *ipos)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < height_d * width_d) {
        unsigned int row = idx / width_d;
        unsigned int col = idx % width_d;
        unsigned int pos = row * width_d + col;

        float value = target[pos];
        if (!isnan(value) && !isinf(value)) {
            int position = atomicAdd(ipos, 1);
            filtered[position] = value;
        }
    }
}

__global__ void process_pixels_STEEP(Endmember *hotCandidates_d, Endmember *coldCandidates_d, int *indexes_d, float *ndvi_d, float *surf_temp_d, float *albedo_d, float *net_radiation_d, float *soil_heat_d, float *ho_d, float ndviQuartileLow, float ndviQuartileHigh, float tsQuartileLow, float tsQuartileMid, float tsQuartileHigh, float albedoQuartileLow, float albedoQuartileMid, float albedoQuartileHigh)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < width_d * height_d) {
        unsigned int row = idx / width_d;
        unsigned int col = idx % width_d;
        unsigned int pos = row * width_d + col;

        ho_d[pos] = net_radiation_d[pos] - soil_heat_d[pos];

        bool hotNDVI = !isnan(ndvi_d[pos]) && ndvi_d[pos] > 0.10 && ndvi_d[pos] < ndviQuartileLow;
        bool hotAlbedo = !isnan(albedo_d[pos]) && albedo_d[pos] > albedoQuartileMid && albedo_d[pos] < albedoQuartileHigh;
        bool hotTS = !isnan(surf_temp_d[pos]) && surf_temp_d[pos] > tsQuartileMid && surf_temp_d[pos] < tsQuartileHigh;

        bool coldNDVI = !isnan(ndvi_d[pos]) && ndvi_d[pos] > ndviQuartileHigh;
        bool coldAlbedo = !isnan(albedo_d[pos]) && albedo_d[pos] > albedoQuartileLow && albedo_d[pos] < albedoQuartileMid;
        bool coldTS = !isnan(surf_temp_d[pos]) && surf_temp_d[pos] < tsQuartileLow;

        if (hotAlbedo && hotNDVI && hotTS) {
            int ih = atomicAdd(&indexes_d[0], 1);
            hotCandidates_d[ih] = Endmember(ndvi_d[pos], surf_temp_d[pos], row, col);
        }

        if (coldNDVI && coldAlbedo && coldTS) {
            int ic = atomicAdd(&indexes_d[1], 1);
            coldCandidates_d[ic] = Endmember(ndvi_d[pos], surf_temp_d[pos], row, col);
        }
    }
}

__global__ void process_pixels_ASEBAL(Endmember *hotCandidates_d, Endmember *coldCandidates_d, int *indexes_d, float *ndvi_d, float *surf_temp_d, float *albedo_d, float *net_radiation_d, float *soil_heat_d, float *ho_d, float ndviHOTQuartile, float ndviCOLDQuartile, float tsHOTQuartile, float tsCOLDQuartile, float albedoHOTQuartile, float albedoCOLDQuartile)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < width_d * height_d) {
        unsigned int row = idx / width_d;
        unsigned int col = idx % width_d;
        unsigned int pos = row * width_d + col;

        ho_d[pos] = net_radiation_d[pos] - soil_heat_d[pos];

        bool hotNDVI = !isnan(ndvi_d[pos]) && ndvi_d[pos] > 0.10 && ndvi_d[pos] < ndviHOTQuartile;
        bool hotAlbedo = !isnan(albedo_d[pos]) && albedo_d[pos] > albedoHOTQuartile;
        bool hotTS = !isnan(surf_temp_d[pos]) && surf_temp_d[pos] > tsHOTQuartile;

        bool coldNDVI = !isnan(ndvi_d[pos]) && ndvi_d[pos] > ndviCOLDQuartile;
        bool coldAlbedo = !isnan(albedo_d[pos]) && albedo_d[pos] < albedoCOLDQuartile;
        bool coldTS = !isnan(surf_temp_d[pos]) && surf_temp_d[pos] < tsCOLDQuartile;

        if (hotAlbedo && hotNDVI && hotTS) {
            int ih = atomicAdd(&indexes_d[0], 1);
            hotCandidates_d[ih] = Endmember(ndvi_d[pos], surf_temp_d[pos], row, col);
        }

        if (coldNDVI && coldAlbedo && coldTS) {
            int ic = atomicAdd(&indexes_d[1], 1);
            coldCandidates_d[ic] = Endmember(ndvi_d[pos], surf_temp_d[pos], row, col);
        }
    }
}
