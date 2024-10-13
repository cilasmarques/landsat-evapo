#pragma once

#include "constants.h"
#include "cuda_utils.h"

/**
 * @brief  Compute the LAI of the bands.
 *
 * @param reflectance_nir_d  The NIR reflectance.
 * @param reflectance_red_d  The red reflectance.
 * @param lai_d  The LAI.
 * @param width_band  The width of the bands.
 * @param height_band  The height of the bands.
 */
__global__ void lai_kernel(float *reflectance_nir_d, float *reflectance_red_d, float *lai_d, int width_band, int height_band);

/**
 * @brief  Compute the ENV of the bands.
 *
 * @param lai_d  The LAI.
 * @param ndvi_d  The NDVI.
 * @param env_d  The ENV.
 * @param width_band  The width of the bands.
 * @param height_band  The height of the bands.
 */
__global__ void enb_kernel(float *lai_d, float *ndvi_d, float *enb_d, int width, int height);

/**
 * @brief  Compute the EO of the bands.
 *
 * @param lai_d  The LAI.
 * @param ndvi_d  The NDVI.
 * @param eo_d  The EO.
 * @param width_band  The width of the bands.
 * @param height_band  The height of the bands.
 */
__global__ void eo_kernel(float *lai_d, float *ndvi_d, float *eo_d, int width_band, int height_band);

/**
 * @brief  Compute the rah correction cycle. (STEEP algorithm)
 *
 * @param surface_temperature_pointer  Surface temperature
 * @param d0_pointer  Zero plane displacement height
 * @param kb1_pointer  KB-1 stability parameter
 * @param zom_pointer  Roughness length for momentum
 * @param ustar_pointer  Ustar pointer
 * @param rah_pointer  Rah pointer
 * @param H_pointer  Sensible heat flux
 * @param a  Coefficient a
 * @param b  Coefficient b
 * @param height  Height of the input data
 * @param width  Width of the input data
 */
__global__ void rah_correction_cycle_STEEP(float *surface_temperature_pointer, float *d0_pointer, float *kb1_pointer, float *zom_pointer, float *ustar_pointer, float *rah_pointer, float *H_pointer, float a, float b, int height, int width);

/**
 * @brief  Compute the rah correction cycle. (STEEP algorithm)
 *
 * @param surface_temperature_pointer  Surface temperature
 * @param kb1_pointer  KB-1 stability parameter
 * @param zom_pointer  Roughness length for momentum
 * @param ustar_pointer  Ustar pointer
 * @param rah_pointer  Rah pointer
 * @param H_pointer  Sensible heat flux
 * @param a  Coefficient a
 * @param b  Coefficient b
 * @param u200 U200
 * @param height  Height of the input data
 * @param width  Width of the input data
 */
__global__ void rah_correction_cycle_ASEBAL(float *surface_temperature_pointer, float *kb1_pointer, float *zom_pointer, float *ustar_pointer, float *rah_pointer, float *H_pointer, float a, float b, float u200, int height, int width);

// ==============================================
// Tensor
// ==============================================

/**
 * @brief Set the invalid values of the radiance, reflectance and albedo to NaN.
 *
 * @param albedo_d The albedo array.
 * @param radiance_blue_d The blue radiance array.
 * @param radiance_green_d The green radiance array.
 * @param radiance_red_d The red radiance array.
 * @param radiance_nir_d The NIR radiance array.
 * @param radiance_swir1_d The SWIR1 radiance array.
 * @param radiance_termal_d The termal radiance array.
 * @param radiance_swir2_d The SWIR2 radiance array.
 * @param reflectance_blue_d The blue reflectance array.
 * @param reflectance_green_d The green reflectance array.
 * @param reflectance_red_d The red reflectance array.
 * @param reflectance_nir_d The NIR reflectance array.
 * @param reflectance_swir1_d The SWIR1 reflectance array.
 * @param reflectance_termal_d The termal reflectance array.
 * @param reflectance_swir2_d The SWIR2 reflectance array.
 * @param width The width of the arrays.
 * @param height The height of the arrays.
 */
__global__ void invalid_rad_ref_kernel(float *albedo_d, float *radiance_blue_d, float *radiance_green_d, float *radiance_red_d, float *radiance_nir_d, float *radiance_swir1_d, float *radiance_termal_d, float *radiance_swir2_d, float *reflectance_blue_d, float *reflectance_green_d, float *reflectance_red_d, float *reflectance_nir_d, float *reflectance_swir1_d, float *reflectance_termal_d, float *reflectance_swir2_d, int width, int height);
