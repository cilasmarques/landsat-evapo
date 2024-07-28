#pragma once

#include "constants.h"
#include "cuda_utils.h"

__global__ void rad_kernel(float *band_blue_d, float *band_green_d, float *band_red_d, float *band_nir_d, float *band_swir1_d, float *band_termal_d, float *band_swir2_d,
                           float *radiance_blue_d, float *radiance_green_d, float *radiance_red_d, float *radiance_nir_d, float *radiance_swir1_d, float *radiance_termal_d, float *radiance_swir2_d,
                           float *rad_add_d, float *rad_mult_d, int width, int height);

__global__ void ref_kernel(float *radiance_blue_d, float *radiance_green_d, float *radiance_red_d, float *radiance_nir_d, float *radiance_swir1_d, float *radiance_termal_d, float *radiance_swir2_d,
                           float *reflectance_blue_d, float *reflectance_green_d, float *reflectance_red_d, float *reflectance_nir_d, float *reflectance_swir1_d, float *reflectance_termal_d, float *reflectance_swir2_d,
                           float *ref_add_d, float *ref_mult_d, float sin_sun, int width, int height);

__global__ void albedo_kernel(float *reflectance_blue_d, float *reflectance_green_d, float *reflectance_red_d, float *reflectance_nir_d, float *reflectance_swir1_d, float *reflectance_swir2_d,
                              float *tal_d, float *albedo_d, float *ref_w_coeff_d, int width, int height);

/**
 * @brief  Compute the rah correction cycle. (STEEP algorithm)
 *
 * @param surface_temperature_pointer  Surface temperature
 * @param d0_pointer  Zero plane displacement height
 * @param kb1_pointer  KB-1 stability parameter
 * @param zom_pointer  Roughness length for momentum
 * @param ustarR_pointer  Ustar pointer for reading
 * @param ustarW_pointer  Ustar pointer for writing
 * @param rahR_pointer  Rah pointer for reading
 * @param rahWL_pointer  Rah pointer for writing
 * @param H_pointer  Sensible heat flux
 * @param a  Coefficient a
 * @param b  Coefficient b
 * @param height  Height of the input data
 * @param width  Width of the input data
 */
__global__ void rah_correction_cycle_STEEP(float *surface_temperature_pointer, float *d0_pointer, float *kb1_pointer, float *zom_pointer, float *ustarR_pointer,
                                           float *ustarW_pointer, float *rahR_pointer, float *rahW_pointer, float *H_pointer, double a, double b, int height,
                                           int width);
