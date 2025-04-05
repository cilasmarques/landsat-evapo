#include "kernels.cuh"

__device__ int hotEndmemberLine_d;
__device__ int hotEndmemberCol_d;
__device__ int coldEndmemberLine_d;
__device__ int coldEndmemberCol_d;

__global__ void filter_valid_values(const float *target, float *filtered, int *ipos)
{
    unsigned int pos = threadIdx.x + blockIdx.x * blockDim.x;

    if (pos < height_d * width_d) {
        float value = target[pos];
        if (!isnan(value) && !isinf(value)) {
            int position = atomicAdd(ipos, 1);
            filtered[position] = value;
        }
    }
}

__global__ void process_pixels_STEEP(Endmember *hotCandidates_d, Endmember *coldCandidates_d, int *indexes_d, float *ndvi_d, float *surface_temperature_d, float *albedo_d, float *net_radiation_d, float *soil_heat_d, float ndviQuartileLow, float ndviQuartileHigh, float tsQuartileLow, float tsQuartileMid, float tsQuartileHigh, float albedoQuartileLow, float albedoQuartileMid, float albedoQuartileHigh)
{
    unsigned int pos = threadIdx.x + blockIdx.x * blockDim.x;

    if (pos < width_d * height_d) {
        bool hotNDVI = !isnan(ndvi_d[pos]) && ndvi_d[pos] > 0.10f && ndvi_d[pos] < ndviQuartileLow;
        bool hotAlbedo = !isnan(albedo_d[pos]) && albedo_d[pos] > albedoQuartileMid && albedo_d[pos] < albedoQuartileHigh;
        bool hotTS = !isnan(surface_temperature_d[pos]) && surface_temperature_d[pos] > tsQuartileMid && surface_temperature_d[pos] < tsQuartileHigh;

        bool coldNDVI = !isnan(ndvi_d[pos]) && ndvi_d[pos] > ndviQuartileHigh;
        bool coldAlbedo = !isnan(albedo_d[pos]) && albedo_d[pos] > albedoQuartileLow && albedo_d[pos] < albedoQuartileMid;
        bool coldTS = !isnan(surface_temperature_d[pos]) && surface_temperature_d[pos] < tsQuartileLow;

        if (hotAlbedo && hotNDVI && hotTS) {
            unsigned int row = pos / width_d;
            unsigned int col = pos % width_d;
            int ih = atomicAdd(&indexes_d[0], 1);
            hotCandidates_d[ih] = Endmember(ndvi_d[pos], surface_temperature_d[pos], row, col);
        }

        if (coldNDVI && coldAlbedo && coldTS) {
            unsigned int row = pos / width_d;
            unsigned int col = pos % width_d;
            int ic = atomicAdd(&indexes_d[1], 1);
            coldCandidates_d[ic] = Endmember(ndvi_d[pos], surface_temperature_d[pos], row, col);
        }
    }
}

__global__ void process_pixels_ASEBAL(Endmember *hotCandidates_d, Endmember *coldCandidates_d, int *indexes_d, float *ndvi_d, float *surface_temperature_d, float *albedo_d, float *net_radiation_d, float *soil_heat_d, float ndviHOTQuartile, float ndviCOLDQuartile, float tsHOTQuartile, float tsCOLDQuartile, float albedoHOTQuartile, float albedoCOLDQuartile)
{
    unsigned int pos = threadIdx.x + blockIdx.x * blockDim.x;

    if (pos < width_d * height_d) {
        bool hotNDVI = !isnan(ndvi_d[pos]) && ndvi_d[pos] > 0.10f && ndvi_d[pos] < ndviHOTQuartile;
        bool hotAlbedo = !isnan(albedo_d[pos]) && albedo_d[pos] > albedoHOTQuartile;
        bool hotTS = !isnan(surface_temperature_d[pos]) && surface_temperature_d[pos] > tsHOTQuartile;

        bool coldNDVI = !isnan(ndvi_d[pos]) && ndvi_d[pos] > ndviCOLDQuartile;
        bool coldAlbedo = !isnan(albedo_d[pos]) && albedo_d[pos] < albedoCOLDQuartile;
        bool coldTS = !isnan(surface_temperature_d[pos]) && surface_temperature_d[pos] < tsCOLDQuartile;

        if (hotAlbedo && hotNDVI && hotTS) {
            unsigned int row = pos / width_d;
            unsigned int col = pos % width_d;
            int ih = atomicAdd(&indexes_d[0], 1);
            hotCandidates_d[ih] = Endmember(ndvi_d[pos], surface_temperature_d[pos], row, col);
        }

        if (coldNDVI && coldAlbedo && coldTS) {
            unsigned int row = pos / width_d;
            unsigned int col = pos % width_d;
            int ic = atomicAdd(&indexes_d[1], 1);
            coldCandidates_d[ic] = Endmember(ndvi_d[pos], surface_temperature_d[pos], row, col);
        }
    }
}
