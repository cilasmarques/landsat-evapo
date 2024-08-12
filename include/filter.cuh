#pragma once

#include "candidate.h"

__global__ void process_pixels(Candidate *hotCandidates, Candidate *coldCandidates, int *d_indexes,
                               float *ndvi, float *surface_temperature, float *albedo, float *net_radiation, float *soil_heat, float *ho,
                               float ndviQuartileLow, float ndviQuartileHigh, float tsQuartileLow, float tsQuartileMid, float tsQuartileHigh,
                               float albedoQuartileLow, float albedoQuartileMid, float albedoQuartileHigh, int height_band, int width_band);
