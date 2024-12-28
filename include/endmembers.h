#pragma once

#include "utils.h"
#include "cuda_utils.h"
#include "constants.h"
#include "candidate.h"

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
void get_quartiles_cuda(float *d_target, float *v_quartile, int height_band, int width_band, float first_interval, float middle_interval, float last_interval, int blocks_num, int threads_num);

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
 * @param hot_pixel: Hot pixel pointer.
 * @param cold_pixel: Cold pixel pointer.
 * @param height_band: Band height.
 * @param width_band: Band width.
 *
 * @retval Candidate
 */
string getEndmembersSTEEP(float *d_ndvi, float *d_surface_temperature, float *albedo,
                          float *d_net_radiation, float *d_soil_heat, int blocks_num, int threads_num,
                          Candidate &hot_pixel, Candidate &cold_pixel, int height_band, int width_band);

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
 * @param hot_pixel: Hot pixel pointer.
 * @param cold_pixel: Cold pixel pointer.
 * @param height_band: Band height.
 * @param width_band: Band width.
 *
 * @retval Candidate
 */
string getEndmembersASEBAL(float *d_ndvi, float *d_surface_temperature, float *albedo,
                           float *d_net_radiation, float *d_soil_heat, int blocks_num, int threads_num,
                           Candidate &hot_pixel, Candidate &cold_pixel, int height_band, int width_band);
