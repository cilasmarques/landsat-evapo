#pragma once

#include "constants.h"

/**
 * @brief  Compute the rah correction cycle. (STEEP algorithm)
 *
 * @param  start_line: The start line of the band.
 * @param  end_line: The end line of the band.
 * @param  width_band: The width of the band.
 * @param  a: The a parameter.
 * @param  b: The b parameter.
 * @param  surface_temperature_pointer: The surface temperature vector.
 * @param  d0_pointer: The d0 vector.
 * @param  aerodynamic_resistance_previous: The aerodynamic resistance vector of the previous iteration.
 * @param  ustar_previous: The ustar vector of the previous iteration.
 * @param  zom_pointer: The zom vector.
 * @param  kb1_pointer: The kb1 vector.
 * @param  sensible_heat_flux_pointer: The sensible heat flux vector.
 * @param  ustar_pointer: The ustar vector.
 * @param  aerodynamic_resistance_pointer: The aerodynamic resistance vector.
 */
void rah_correction_cycle_STEEP(int start_line, int end_line, int width_band, float a, float b, float *surface_temperature_pointer, float *d0_pointer, float *zom_pointer, float *kb1_pointer, float *sensible_heat_flux_pointer, float *ustar_pointer, float *aerodynamic_resistance_pointer);

/**
 * @brief  Compute the rah correction cycle. (STEEP algorithm)
 *
 * @param  start_line: The start line of the band.
 * @param  end_line: The end line of the band.
 * @param  width_band: The width of the band.
 * @param  a: The a parameter.
 * @param  b: The b parameter.
 * @param  surface_temperature_pointer: The surface temperature vector.
 * @param  d0_pointer: The d0 vector.
 * @param  aerodynamic_resistance_previous: The aerodynamic resistance vector of the previous iteration.
 * @param  ustar_previous: The ustar vector of the previous iteration.
 * @param  zom_pointer: The zom vector.
 * @param  kb1_pointer: The kb1 vector.
 * @param  sensible_heat_flux_pointer: The sensible heat flux vector.
 * @param  ustar_pointer: The ustar vector.
 * @param  u200: The u200 parameter.
 * @param  aerodynamic_resistance_pointer: The aerodynamic resistance vector.
 */
void rah_correction_cycle_ASEBAL(int start_line, int end_line, int width_band, float a, float b, float *surface_temperature_pointer, float *zom_pointer, float *kb1_pointer, float *sensible_heat_flux_pointer, float *ustar_pointer, float u200, float *aerodynamic_resistance_pointer);

