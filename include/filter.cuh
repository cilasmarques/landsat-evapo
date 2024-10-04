#pragma once

#include "candidate.h"

/**
 * @brief Filter values that are not NaN or Inf.
 * 
 * @param target The target array to filter.
 * @param filtered The filtered array.
 * @param height_band The height of the target array.
 * @param width_band The width of the target array.
 * @param pos The position of the filtered array.
 */
__global__ void filter_valid_values(const float *target, float *filtered, int height_band, int width_band, int *pos);

/**
 * @brief Process the pixels of the target arrays and store the candidates in the hot and cold arrays.
 *
 * @param hotCandidates The hot candidates array.
 * @param coldCandidates The cold candidates array.
 * @param d_indexes The indexes of the hot and cold arrays.
 * @param ndvi The NDVI array.
 * @param surface_temperature The surface temperature array.
 * @param albedo The albedo array.
 * @param net_radiation The net radiation array.
 * @param soil_heat The soil heat array.
 * @param ho The ho array.
 * @param ndviQuartileLow The NDVI low quartile.
 * @param ndviQuartileHigh The NDVI high quartile.
 * @param tsQuartileLow The surface temperature low quartile.
 * @param tsQuartileMid The surface temperature mid quartile.
 * @param tsQuartileHigh The surface temperature high quartile.
 * @param albedoQuartileLow The albedo low quartile.
 * @param albedoQuartileMid The albedo mid quartile.
 * @param albedoQuartileHigh The albedo high quartile.
 * @param height_band The height of the target arrays.
 * @param width_band The width of the target arrays.
 */
__global__ void process_pixels(Candidate *hotCandidates, Candidate *coldCandidates, int *d_indexes,
                               float *ndvi, float *surface_temperature, float *albedo, float *net_radiation, float *soil_heat, float *ho,
                               float ndviQuartileLow, float ndviQuartileHigh, float tsQuartileLow, float tsQuartileMid, float tsQuartileHigh,
                               float albedoQuartileLow, float albedoQuartileMid, float albedoQuartileHigh, int height_band, int width_band);
