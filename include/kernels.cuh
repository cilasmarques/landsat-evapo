#pragma once

#include "constants.h"
#include "cuda_utils.h"

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
