#include "filter.cuh"

__global__ void process_pixels(Candidate *hotCandidates, Candidate *coldCandidates, int *d_hot_index, int *d_cold_index,
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

        bool hotNDVI = !isnan(ndvi[pos]) && ndvi[pos] > 0.10 && ndvi[pos] < ndviQuartileLow;
        bool hotAlbedo = !isnan(albedo[pos]) && albedo[pos] > albedoQuartileMid && albedo[pos] < albedoQuartileHigh;
        bool hotTS = !isnan(surface_temperature[pos]) && surface_temperature[pos] > tsQuartileMid && surface_temperature[pos] < tsQuartileHigh;

        bool coldNDVI = !isnan(ndvi[pos]) && ndvi[pos] > ndviQuartileHigh;
        bool coldAlbedo = !isnan(surface_temperature[pos]) && albedo[pos] > albedoQuartileLow && albedo[pos] < albedoQuartileMid;
        bool coldTS = !isnan(albedo[pos]) && surface_temperature[pos] < tsQuartileLow;

        if (hotAlbedo && hotNDVI && hotTS)
        {
            hotCandidates[atomicAdd(d_hot_index, 1)] = Candidate(ndvi[pos], surface_temperature[pos], net_radiation[pos], soil_heat[pos], ho[pos], row, col);
        }

        if (coldNDVI && coldAlbedo && coldTS)
        {
            coldCandidates[atomicAdd(d_cold_index, 1)] = Candidate(ndvi[pos], surface_temperature[pos], net_radiation[pos], soil_heat[pos], ho[pos], row, col);
        }
    }
}
