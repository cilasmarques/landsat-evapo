#include "filter.cuh"

__global__ void filter_valid_values(const float *target, float *filtered, int height_band, int width_band, int *pos) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = height_band * width_band;

    if (idx < size) {
        float value = target[idx];
        if (!isnan(value) && !isinf(value)) {
            int position = atomicAdd(pos, 1);
            filtered[position] = value;
        }
    }
}

__global__ void process_pixels_STEEP(Candidate *hotCandidates, Candidate *coldCandidates, int *d_indexes,
                               float *ndvi, float *surface_temperature, float *albedo, float *net_radiation, float *soil_heat, float *ho,
                               float ndviQuartileLow, float ndviQuartileHigh, float tsQuartileLow, float tsQuartileMid, float tsQuartileHigh,
                               float albedoQuartileLow, float albedoQuartileMid, float albedoQuartileHigh, int height_band, int width_band)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Map 1D position to 2D grid
    unsigned int row = idx / width_band;
    unsigned int col = idx % width_band;

    if (idx < width_band * height_band)
    {
        unsigned int pos = row * width_band + col;

        ho[pos] = net_radiation[pos] - soil_heat[pos];

        bool hotNDVI = !isnan(ndvi[pos]) && ndvi[pos] > 0.10 && ndvi[pos] < ndviQuartileLow;
        bool hotAlbedo = !isnan(albedo[pos]) && albedo[pos] > albedoQuartileMid && albedo[pos] < albedoQuartileHigh;
        bool hotTS = !isnan(surface_temperature[pos]) && surface_temperature[pos] > tsQuartileMid && surface_temperature[pos] < tsQuartileHigh;

        bool coldNDVI = !isnan(ndvi[pos]) && ndvi[pos] > ndviQuartileHigh;
        bool coldAlbedo = !isnan(surface_temperature[pos]) && albedo[pos] > albedoQuartileLow && albedo[pos] < albedoQuartileMid;
        bool coldTS = !isnan(albedo[pos]) && surface_temperature[pos] < tsQuartileLow;

        if (hotAlbedo && hotNDVI && hotTS)
        {
            int ih = atomicAdd(&d_indexes[0],1);
            hotCandidates[ih] = Candidate(ndvi[pos], surface_temperature[pos], net_radiation[pos], soil_heat[pos], ho[pos], row, col);
        }

        if (coldNDVI && coldAlbedo && coldTS)
        {
            int ic = atomicAdd(&d_indexes[1],1);
            coldCandidates[ic] = Candidate(ndvi[pos], surface_temperature[pos], net_radiation[pos], soil_heat[pos], ho[pos], row, col);
        }
    }
}

__global__ void process_pixels_ASEBAL(Candidate *hotCandidates, Candidate *coldCandidates, int *d_indexes,
                               float *ndvi, float *surface_temperature, float *albedo, float *net_radiation, float *soil_heat, float *ho,
                               float ndviHOTQuartile, float ndviCOLDQuartile, float tsHOTQuartile, float tsCOLDQuartile,
                               float albedoHOTQuartile, float albedoCOLDQuartile, int height_band, int width_band)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Map 1D position to 2D grid
    unsigned int row = idx / width_band;
    unsigned int col = idx % width_band;

    if (idx < width_band * height_band)
    {
        unsigned int pos = row * width_band + col;

        ho[pos] = net_radiation[pos] - soil_heat[pos];

        bool hotNDVI = !isnan(ndvi[pos]) && ndvi[pos] > 0.10 && ndvi[pos] < ndviHOTQuartile;
        bool hotAlbedo = !isnan(albedo[pos]) && albedo[pos] > albedoHOTQuartile;
        bool hotTS = !isnan(surface_temperature[pos]) && surface_temperature[pos] > tsHOTQuartile;

        bool coldNDVI = !isnan(ndvi[pos]) && ndvi[pos] > ndviCOLDQuartile;
        bool coldAlbedo = !isnan(surface_temperature[pos]) && albedo[pos] < albedoCOLDQuartile;
        bool coldTS = !isnan(albedo[pos]) && surface_temperature[pos] < tsCOLDQuartile;

        if (hotAlbedo && hotNDVI && hotTS)
        {
            int ih = atomicAdd(&d_indexes[0],1);
            hotCandidates[ih] = Candidate(ndvi[pos], surface_temperature[pos], net_radiation[pos], soil_heat[pos], ho[pos], row, col);
        }

        if (coldNDVI && coldAlbedo && coldTS)
        {
            int ic = atomicAdd(&d_indexes[1],1);
            coldCandidates[ic] = Candidate(ndvi[pos], surface_temperature[pos], net_radiation[pos], soil_heat[pos], ho[pos], row, col);
        }
    }
}
