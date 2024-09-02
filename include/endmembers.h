#pragma once

#include "utils.h"
#include "cuda_utils.h"
#include "constants.h"
#include "candidate.h"

/**
 * @brief Calculates the four quartiles of a vector. CPU version.
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
 * @brief Calculates the four quartiles of a vector. GPU version.
 *
 * @param d_target: Vector to be calculated the quartiles in device.
 * @param v_quartile: Vector to store the quartiles.
 * @param height_band: Band height.
 * @param width_band: Band width.
 * @param first_interval: First interval.
 * @param middle_interval: Middle interval.
 * @param last_interval: Last interval.
 * @param blocks_num: Number of blocks.
 * @param threads_num: Number of threads.
 *
 * @retval void
 */
void get_quartiles_cuda(float *d_target, float *v_quartile, int height_band, int width_band, float first_interval, float middle_interval, float last_interval,int blocks_num, int threads_num)

/**
 * @brief Get the hot pixel based on the STEPP algorithm. CPU version.
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
 * @brief Get the hot pixel based on the STEPP algorithm. GPU version.
 *
 * @param ndvi: NDVI vector.
 * @param d_ndvi: NDVI vector in device.
 * @param surface_temperature: Surface temperature vector.
 * @param d_surface_temperature: Surface temperature vector in device.
 * @param albedo: Albedo vector.
 * @param d_albedo: Albedo vector in device.
 * @param net_radiation: Net radiation vector.
 * @param d_net_radiation: Net radiation vector in device.
 * @param soil_heat: Soil heat flux vector.
 * @param d_soil_heat: Soil heat flux vector in device.
 * @param blocks_num: Number of blocks.
 * @param threads_num: Number of threads.
 * @param height_band: Band height.
 * @param width_band: Band width.
 *
 * @retval Candidate
 */
pair<Candidate, Candidate> getEndmembersSTEPP(float *ndvi, float *d_ndvi, float *surface_temperature, float *d_surface_temperature, float *albedo, float *d_albedo,
                                              float *net_radiation, float *d_net_radiation, float *soil_heat, float *d_soil_heat,
                                              int blocks_num, int threads_num, int height_band, int width_band);

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
