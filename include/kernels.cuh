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

__global__ void ndvi_kernel(float *band_nir_d, float *band_red_d, float *ndvi_d, int width, int height);

__global__ void pai_kernel(float *band_nir_d, float *band_red_d, float *pai_d, int width, int height);

__global__ void lai_kernel(float *reflectance_nir_d, float *reflectance_red_d, float *lai_d, int width_band, int height_band);

__global__ void evi_kernel(float *reflectance_nir_d, float *reflectance_red_d, float *reflectance_blue_d, float *evi_d, int width_band, int height_band);

__global__ void enb_kernel(float *lai_d, float *ndvi_d, float *enb_d, int width, int height);

__global__ void eo_kernel(float *lai_d, float *ndvi_d, float *eo_d, int width_band, int height_band);

__global__ void ea_kernel(float *tal_d, float *ea_d, int width_band, int height_band);

__global__ void surface_temperature_kernel(float *enb_d, float *radiance_termal_d, float *surface_temperature_d, float k1, float k2, int width_band, int height_band);

__global__ void short_wave_radiation_kernel(float *tal_d, float *short_wave_radiation_d, float sun_elevation, float distance_earth_sun, float pi, int width_band, int height_band);

__global__ void large_wave_radiation_surface_kernel(float *surface_temperature_d, float *eo_d, float *large_wave_radiation_surface_d, int width_band, int height_band);

__global__ void large_wave_radiation_atmosphere_kernel(float *ea_d, float *large_wave_radiation_atmosphere_d, float temperature, int width_band, int height_band);

__global__ void net_radiation_kernel(float *short_wave_radiation_d, float *albedo_d, float *large_wave_radiation_atmosphere_d, float *large_wave_radiation_surface_d, float *eo_d, float *net_radiation_d, int width_band, int height_band);

__global__ void soil_heat_kernel(float *ndvi_d, float *albedo_d, float *surface_temperature_d, float *net_radiation_d, float *soil_heat_d, int width_band, int height_band);

__global__ void d0_kernel(float *pai_d, float *d0_d, float CD1, float HGHT, int width_band, int height_band);

__global__ void zom_kernel(float *d0_d, float *pai_d, float *zom_d, float A_ZOM, float B_ZOM, int width_band, int height_band);

__global__ void ustar_kernel(float *zom_d, float *d0_d, float *ustarR_d, float u10, int width_band, int height_band);

__global__ void kb_kernel(float *zom_d, float *ustarR_d, float *pai_d, float *kb1_d, float *ndvi_d, int width_band, int height_band, float ndvi_max, float ndvi_min);

__global__ void aerodynamic_resistance_kernel(float *zom_d, float *d0_d, float *ustarR_d, float *kb1_d, float *rahR_d, int width_band, int height_band);

__global__ void sensible_heat_flux_kernel(float *surface_temperature_d, float *rahR_d, float *net_radiation_d, float *soil_heat_d, float *sensible_heat_flux_d, float a, float b, int width_band, int height_band);

__global__ void latent_heat_flux_kernel(float *net_radiation_d, float *soil_heat_d, float *sensible_heat_flux_d, float *latent_heat_flux_d, int width_band, int height_band);

__global__ void net_radiation_24h_kernel(float *albedo_d, float Rs24h, float Ra24h, float *net_radiation_24h_d, int width_band, int height_band);

__global__ void evapotranspiration_fraction_kernel(float *net_radiation_d, float *soil_heat_d, float *latent_heat_flux_d, float *evapotranspiration_fraction_d, int width_band, int height_band);

__global__ void sensible_heat_flux_24h_kernel(float *net_radiation_24h_d, float *evapotranspiration_fraction_d, float *sensible_heat_flux_24h_d, int width_band, int height_band);

__global__ void latent_heat_flux_24h_kernel(float *net_radiation_24h_d, float *evapotranspiration_fraction_d, float *latent_heat_flux_24h_d, int width_band, int height_band);

__global__ void evapotranspiration_24h_kernel(float *latent_heat_flux_24h_d, float *evapotranspiration_24h_d, float v7_max, float v7_min, int width_band, int height_band);

__global__ void evapotranspiration_kernel(float *net_radiation_24h_d, float *evapotranspiration_fraction_d, float *evapotranspiration_d, int width_band, int height_band);

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
                                           float *ustarW_pointer, float *rahR_pointer, float *rahW_pointer, float *H_pointer, float a, float b, int height,
                                           int width);
