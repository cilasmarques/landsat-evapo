#pragma once

#include "utils.h"
#include "constants.h"
#include "parameters.h"
#include "candidate.h"

/**
 * @brief  Compute the H0
 *
 * @param  net_radiation: Net radiation vector.
 * @param  soil_heat_flux: Soil heat flux vector.
 * @param  height_band: Band height.
 * @param  width_band: Band width.
 * @param  ho: H0 vector.
 *
 * @retval void
 */
void compute_H0(float *net_radiation, float *soil_heat_flux, int height_band, int width_band, float *ho);

/**
 * @brief Calculates the four quartiles of a vector.
 *
 * @param target: Vector to be calculated the quartiles.
 * @param v_quartile: Vector to store the quartiles.
 * @param height_band: Band height.
 * @param width_band: Band width.
 * @param first_interval: First interval.
 * @param middle_interval: Middle interval.
 * @param last_interval: Last interval.
 *
 * @retval void
 */
void get_quartiles(float *target, float *v_quartile, int height_band, int width_band, float first_interval, float middle_interval, float last_interval);

/**
 * @brief Get the hot pixel based on the STEPP algorithm.
 *
 * @param ndvi: NDVI vector.
 * @param surface_temperature: Surface temperature vector.
 * @param albedo: Albedo vector.
 * @param net_radiation: Net radiation vector.
 * @param soil_heat: Soil heat flux vector.
 * @param height_band: Band height.
 * @param width_band: Band width.
 *
 * @retval Candidate
 */
pair<Candidate, Candidate> getEndmembersSTEPP(float *ndvi, float *surface_temperature, float *albedo, float *net_radiation, float *soil_heat, int height_band, int width_band);

/**
 * @brief Get the hot and cold pixels based on the ASEBAL algorithm.
 *
 * @param ndvi_vector: NDVI vector.
 * @param surface_temperature_vector: Surface temperature vector.
 * @param albedo_vector: Albedo vector.
 * @param net_radiation_vector: Net radiation vector.
 * @param soil_heat_vector: Soil heat flux vector.
 * @param height_band: Band height.
 * @param width_band: Band width.
 *
 * @retval Candidate
 */
pair<Candidate, Candidate> getEndmembersASEBAL(float *ndvi, float *surface_temperature, float *albedo, float *net_radiation, float *soil_heat, int height_band, int width_band);
