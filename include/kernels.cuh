#pragma once

#include "constants.h"
#include "cuda_utils.h"
#include "surfaceData.cuh"

extern __device__ int width_d;
extern __device__ int height_d;

extern __device__ int hotEndmemberLine_d;
extern __device__ int hotEndmemberCol_d;
extern __device__ int coldEndmemberLine_d;
extern __device__ int coldEndmemberCol_d;

__global__ void NAN_kernel(float *pointer_d);

/**
 * @brief  Compute the LAI of the bands.
 *
 * @param reflectance_nir_d  The NIR reflectance.
 * @param reflectance_red_d  The red reflectance.
 * @param lai_d  The LAI.
 */
__global__ void lai_kernel(float *reflectance_nir_d, float *reflectance_red_d, float *lai_d);

/**
 * @brief  Compute the ENV of the bands.
 *
 * @param lai_d  The LAI.
 * @param ndvi_d  The NDVI.
 * @param env_d  The ENV.
 */
__global__ void enb_kernel(float *lai_d, float *ndvi_d, float *enb_d);

/**
 * @brief  Compute the EO of the bands.
 *
 * @param lai_d  The LAI.
 * @param ndvi_d  The NDVI.
 * @param eo_d  The EO.
 */
__global__ void eo_kernel(float *lai_d, float *ndvi_d, float *eo_d);

/**
 * @brief  Compute the surface temperature of the bands.
 *
 * @param enb_d  The ENB.
 * @param radiance_termal_d  The termal radiance.
 * @param surface_temperature_d  The surface temperature.
 * @param k1  The K1 constant.
 * @param k2  The K2 constant.
 */
__global__ void surface_temperature_kernel(float *enb_d, float *radiance_termal_d, float *surface_temperature_d, float k1, float k2);

/**
 * @brief  Compute the rah correction cycle. (STEEP algorithm)
 *
 * @param net_radiation_d  Net radiation
 * @param soil_heat_flux_d  Soil heat flux
 * @param ndvi_d  NDVI
 * @param surf_temp_d  Surface temperature
 * @param d0_d  Zero plane displacement height_d
 * @param kb1_d  KB-1 stability parameter
 * @param zom_d  Roughness length for momentum
 * @param ustar_d  Ustar pointer
 * @param rah_d  Rah pointer
 * @param H_d  Sensible heat flux
 * @param ndvi_max  NDVI max value
 * @param ndvi_min  NDVI min value
 */
__global__ void rah_correction_cycle_STEEP(float *net_radiation_d, float *soil_heat_flux_d, float *ndvi_d, float *surf_temp_d, float *d0_d, float *kb1_d, float *zom_d, float *ustar_d, float *rah_d, float *H_d, float *a_d, float *b_d, float ndvi_max, float ndvi_min);

/**
 * @brief  Compute the rah correction cycle. (STEEP algorithm)
 *
 * @param net_radiation_d  Net radiation
 * @param soil_heat_flux_d  Soil heat flux
 * @param ndvi_d  NDVI
 * @param surf_temp_d  Surface temperature
 * @param kb1_d  KB-1 stability parameter
 * @param zom_d  Roughness length for momentum
 * @param ustar_d  Ustar pointer
 * @param rah_d  Rah pointer
 * @param H_d  Sensible heat flux
 * @param u200 U200
 * @param stop_condition Stop condition
 */
__global__ void rah_correction_cycle_ASEBAL(float *net_radiation_d, float *soil_heat_flux_d, float *ndvi_d, float *surf_temp_d, float *kb1_d, float *zom_d, float *ustar_d, float *rah_d, float *H_d, float *a_d, float *b_d, float u200, int *stop_condition);

/**
 * @brief Filter values that are not NaN or Inf.
 *
 * @param target The target array to filter.
 * @param filtered The filtered array.
 * @param pos The position of the filtered array.
 */
__global__ void filter_valid_values(const float *target, float *filtered, int *pos);

/**
 * @brief Process the pixels of the target arrays and store the candidates in the hot and cold arrays.
 *
 * @param hotCandidates The hot candidates array.
 * @param coldCandidates The cold candidates array.
 * @param indexes_d The indexes of the hot and cold arrays.
 * @param ndvi The NDVI array.
 * @param surface_temperature The surface temperature array.
 * @param albedo The albedo array.
 * @param net_radiation The net radiation array.
 * @param soil_heat The soil heat array.
 * @param ndviQuartileLow The NDVI low quartile.
 * @param ndviQuartileHigh The NDVI high quartile.
 * @param tsQuartileLow The surface temperature low quartile.
 * @param tsQuartileMid The surface temperature mid quartile.
 * @param tsQuartileHigh The surface temperature high quartile.
 * @param albedoQuartileLow The albedo low quartile.
 * @param albedoQuartileMid The albedo mid quartile.
 * @param albedoQuartileHigh The albedo high quartile.
 */
__global__ void process_pixels_STEEP(Endmember *hotCandidates_d, Endmember *coldCandidates_d, int *indexes_d, float *ndvi_d, float *surf_temp_d, float *albedo_d, float *net_radiation_d, float *soil_heat_d, float ndviQuartileLow, float ndviQuartileHigh, float tsQuartileLow, float tsQuartileMid, float tsQuartileHigh, float albedoQuartileLow, float albedoQuartileMid, float albedoQuartileHigh);

/**
 * @brief Process the pixels of the target arrays and store the candidates in the hot and cold arrays.
 *
 * @param hotCandidates The hot candidates array.
 * @param coldCandidates The cold candidates array.
 * @param indexes_d The indexes of the hot and cold arrays.
 * @param ndvi The NDVI array.
 * @param surface_temperature The surface temperature array.
 * @param albedo The albedo array.
 * @param net_radiation The net radiation array.
 * @param soil_heat The soil heat array.
 * @param ndvi1stQuartile The NDVI 1st quartile.
 * @param ndvi4stQuartile The NDVI 4st quartile.
 * @param ts1stQuartile The surface temperature 1st quartile.
 * @param ts3rdQuartile The surface temperature 3rd quartile.
 * @param albedo2ndQuartile The albedo 2nd quartile.
 * @param albedo3rdQuartile The albedo 3rd quartile.
 */
__global__ void process_pixels_ASEBAL(Endmember *hotCandidates_d, Endmember *coldCandidates_d, int *indexes_d, float *ndvi_d, float *surf_temp_d, float *albedo_d, float *net_radiation_d, float *soil_heat_d, float ndviHOTQuartile, float ndviCOLDQuartile, float tsHOTQuartile, float tsCOLDQuartile, float albedoHOTQuartile, float albedoCOLDQuartile);
