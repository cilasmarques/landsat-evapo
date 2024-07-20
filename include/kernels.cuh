#pragma once

#include "constants.h"
#include "cuda_utils.h"

__global__ void rad_kernel(float *band1_d, float *band2_d, float *band3_d, float *band4_d,
                           float *band5_d, float *band6_d, float *band7_d,
                           float *radiance1_d, float *radiance2_d, float *radiance3_d, float *radiance4_d,
                           float *radiance5_d, float *radiance6_d, float *radiance7_d,
                           float grenscale1_d, float brescale1_d,
                           float grenscale2_d, float brescale2_d,
                           float grenscale3_d, float brescale3_d,
                           float grenscale4_d, float brescale4_d,
                           float grenscale5_d, float brescale5_d,
                           float grenscale6_d, float brescale6_d,
                           float grenscale7_d, float brescale7_d,
                           float grenscale8_d,
                           int width, int height);

__global__ void ref_kernel(float sin_sun, 
                           float *radiance1_d, float *radiance2_d, float *radiance3_d, float *radiance4_d, 
                           float *radiance5_d, float *radiance6_d, float *radiance7_d,
                           float *reflectance1_d, float *reflectance2_d, float *reflectance3_d, float *reflectance4_d, 
                           float *reflectance5_d, float *reflectance6_d, float *reflectance7_d,
                           int width, int height);

__global__ void ref_kernel(double PI, float sin_sun, float distance_earth_sun,
                           float esun1, float esun2, float esun3, float esun4,
                           float esun5, float esun6, float esun7,
                           float *radiance1_d, float *radiance2_d, float *radiance3_d, float *radiance4_d,
                           float *radiance5_d, float *radiance6_d, float *radiance7_d,
                           float *reflectance1_d, float *reflectance2_d, float *reflectance3_d, float *reflectance4_d,
                           float *reflectance5_d, float *reflectance6_d, float *reflectance7_d,
                           int width, int height);

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
