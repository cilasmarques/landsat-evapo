#pragma once

#include "constants.h"
#include "cuda_utils.h"

/**
 * @brief  Compute the radiance of the bands.
 *
 * @param band_blue_d  The blue band.
 * @param band_green_d  The green band.
 * @param band_red_d  The red band.
 * @param band_nir_d  The NIR band.
 * @param band_swir1_d  The SWIR1 band.
 * @param band_termal_d  The termal band.
 * @param band_swir2_d  The SWIR2 band.
 * @param radiance_blue_d  The blue radiance.
 * @param radiance_green_d  The green radiance.
 * @param radiance_red_d  The red radiance.
 * @param radiance_nir_d  The NIR radiance.
 * @param radiance_swir1_d  The SWIR1 radiance.
 * @param radiance_termal_d  The termal radiance.
 * @param radiance_swir2_d  The SWIR2 radiance.
 * @param rad_add_d  The radiance add value.
 * @param rad_mult_d  The radiance mult value.
 * @param width  The width of the bands.
 * @param height  The height of the bands.
 */
__global__ void rad_kernel(float *band_blue_d, float *band_green_d, float *band_red_d, float *band_nir_d, float *band_swir1_d, float *band_termal_d, float *band_swir2_d,
                           float *radiance_blue_d, float *radiance_green_d, float *radiance_red_d, float *radiance_nir_d, float *radiance_swir1_d, float *radiance_termal_d, float *radiance_swir2_d,
                           float *rad_add_d, float *rad_mult_d, int width, int height);

/**
 * @brief  Compute the reflectance of the bands.
 *
 * @param radiance_blue_d  The blue radiance.
 * @param radiance_green_d  The green radiance.
 * @param radiance_red_d  The red radiance.
 * @param radiance_nir_d  The NIR radiance.
 * @param radiance_swir1_d  The SWIR1 radiance.
 * @param radiance_termal_d  The termal radiance.
 * @param radiance_swir2_d  The SWIR2 radiance.
 * @param reflectance_blue_d  The blue reflectance.
 * @param reflectance_green_d  The green reflectance.
 * @param reflectance_red_d  The red reflectance.
 * @param reflectance_nir_d  The NIR reflectance.
 * @param reflectance_swir1_d  The SWIR1 reflectance.
 * @param reflectance_termal_d  The termal reflectance.
 * @param reflectance_swir2_d  The SWIR2 reflectance.
 * @param ref_add_d  The reflectance add value.
 * @param ref_mult_d  The reflectance mult value.
 * @param sin_sun  The sin of the sun.
 * @param width  The width of the bands.
 * @param height  The height of the bands.
 */
__global__ void ref_kernel(float *radiance_blue_d, float *radiance_green_d, float *radiance_red_d, float *radiance_nir_d, float *radiance_swir1_d, float *radiance_termal_d, float *radiance_swir2_d,
                           float *reflectance_blue_d, float *reflectance_green_d, float *reflectance_red_d, float *reflectance_nir_d, float *reflectance_swir1_d, float *reflectance_termal_d, float *reflectance_swir2_d,
                           float *ref_add_d, float *ref_mult_d, float sin_sun, int width, int height);

/**
 * @brief  Compute the albedo of the bands.
 *
 * @param reflectance_blue_d  The blue reflectance.
 * @param reflectance_green_d  The green reflectance.
 * @param reflectance_red_d  The red reflectance.
 * @param reflectance_nir_d  The NIR reflectance.
 * @param reflectance_swir1_d  The SWIR1 reflectance.
 * @param reflectance_swir2_d  The SWIR2 reflectance.
 * @param tal_d  The total absorbed radiation.
 * @param albedo_d  The albedo.
 * @param ref_w_coeff_d  The reflectance water coefficient.
 * @param width  The width of the bands.
 * @param height  The height of the bands.
 */
__global__ void albedo_kernel(float *reflectance_blue_d, float *reflectance_green_d, float *reflectance_red_d, float *reflectance_nir_d, float *reflectance_swir1_d, float *reflectance_swir2_d,
                              float *tal_d, float *albedo_d, float *ref_w_coeff_d, int width, int height);

/**
 * @brief  Compute the NDVI of the bands.
 *
 * @param band_nir_d  The NIR band.
 * @param band_red_d  The red band.
 * @param ndvi_d  The NDVI.
 * @param width  The width of the bands.
 * @param height  The height of the bands.
 */
__global__ void ndvi_kernel(float *band_nir_d, float *band_red_d, float *ndvi_d, int width, int height);

/**
 * @brief  Compute the NDWI of the bands.
 *
 * @param band_green_d  The green band.
 * @param band_swir1_d  The SWIR1 band.
 * @param ndwi_d  The NDWI.
 * @param width  The width of the bands.
 * @param height  The height of the bands.
 */
__global__ void pai_kernel(float *band_nir_d, float *band_red_d, float *pai_d, int width, int height);

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
 * @brief  Compute the EVI of the bands.
 *
 * @param reflectance_nir_d  The NIR reflectance.
 * @param reflectance_red_d  The red reflectance.
 * @param reflectance_blue_d  The blue reflectance.
 * @param evi_d  The EVI.
 * @param width_band  The width of the bands.
 * @param height_band  The height of the bands.
 */
__global__ void evi_kernel(float *reflectance_nir_d, float *reflectance_red_d, float *reflectance_blue_d, float *evi_d, int width_band, int height_band);

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
 * @brief  Compute the EA of the bands.
 *
 * @param tal_d  The TAL.
 * @param ea_d  The EA.
 * @param width_band  The width of the bands.
 * @param height_band  The height of the bands.
 */
__global__ void ea_kernel(float *tal_d, float *ea_d, int width_band, int height_band);

/**
 * @brief  Compute the surface temperature of the bands.
 *
 * @param enb_d  The ENB.
 * @param radiance_termal_d  The termal radiance.
 * @param surface_temperature_d  The surface temperature.
 * @param k1  The K1 constant.
 * @param k2  The K2 constant.
 * @param width_band  The width of the bands.
 * @param height_band  The height of the bands.
 */
__global__ void surface_temperature_kernel(float *enb_d, float *radiance_termal_d, float *surface_temperature_d, float k1, float k2, int width_band, int height_band);

/**
 * @brief  Compute the short wave radiation of the bands.
 *
 * @param tal_d  The TAL.
 * @param short_wave_radiation_d  The short wave radiation.
 * @param sun_elevation  The sun elevation.
 * @param distance_earth_sun  The distance between the earth and the sun.
 * @param pi  The pi constant.
 * @param width_band  The width of the bands.
 * @param height_band  The height of the bands.
 */
__global__ void short_wave_radiation_kernel(float *tal_d, float *short_wave_radiation_d, float sun_elevation, float distance_earth_sun, float pi, int width_band, int height_band);

/**
 * @brief  Compute the large wave radiation of the bands.
 *
 * @param surface_temperature_d  The surface temperature.
 * @param eo_d  The EO.
 * @param large_wave_radiation_surface_d  The large wave radiation surface.
 * @param width_band  The width of the bands.
 * @param height_band  The height of the bands.
 */
__global__ void large_wave_radiation_surface_kernel(float *surface_temperature_d, float *eo_d, float *large_wave_radiation_surface_d, int width_band, int height_band);

/**
 * @brief  Compute the large wave radiation of the bands.
 *
 * @param ea_d  The EA.
 * @param large_wave_radiation_atmosphere_d  The large wave radiation atmosphere.
 * @param temperature  The temperature.
 * @param width_band  The width of the bands.
 * @param height_band  The height of the bands.
 */
__global__ void large_wave_radiation_atmosphere_kernel(float *ea_d, float *large_wave_radiation_atmosphere_d, float temperature, int width_band, int height_band);

/**
 * @brief  Compute the net radiation of the bands.
 *
 * @param short_wave_radiation_d  The short wave radiation.
 * @param albedo_d  The albedo.
 * @param large_wave_radiation_atmosphere_d  The large wave radiation atmosphere.
 * @param large_wave_radiation_surface_d  The large wave radiation surface.
 * @param eo_d  The EO.
 * @param net_radiation_d  The net radiation.
 * @param width_band  The width of the bands.
 * @param height_band  The height of the bands.
 */
__global__ void net_radiation_kernel(float *short_wave_radiation_d, float *albedo_d, float *large_wave_radiation_atmosphere_d, float *large_wave_radiation_surface_d, float *eo_d, float *net_radiation_d, int width_band, int height_band);

/**
 * @brief  Compute the soil heat of the bands.
 *
 * @param ndvi_d  The NDVI.
 * @param albedo_d  The albedo.
 * @param surface_temperature_d  The surface temperature.
 * @param net_radiation_d  The net radiation.
 * @param soil_heat_d  The soil heat.
 * @param width_band  The width of the bands.
 * @param height_band  The height of the bands.
 */
__global__ void soil_heat_kernel(float *ndvi_d, float *albedo_d, float *surface_temperature_d, float *net_radiation_d, float *soil_heat_d, int width_band, int height_band);

/**
 * @brief  Compute the zero plane displacement height of the bands.
 *
 * @param pai_d  The PAI.
 * @param d0_d  The D0.
 * @param CD1  The CD1 constant.
 * @param HGHT  The HGHT constant.
 * @param width_band  The width of the bands.
 * @param height_band  The height of the bands.
 */
__global__ void d0_kernel(float *pai_d, float *d0_d, float CD1, float HGHT, int width_band, int height_band);

/**
 * @brief  Compute the roughness length for momentum of the bands.
 *
 * @param d0_d  The D0.
 * @param pai_d  The PAI.
 * @param zom_d  The ZOM.
 * @param A_ZOM  The A_ZOM constant.
 * @param B_ZOM  The B_ZOM constant.
 * @param width_band  The width of the bands.
 * @param height_band  The height of the bands.
 */
__global__ void zom_kernel(float *d0_d, float *pai_d, float *zom_d, float A_ZOM, float B_ZOM, int width_band, int height_band);

/**
 * @brief  Compute the ustar of the bands.
 *
 * @param zom_d  The ZOM.
 * @param d0_d  The D0.
 * @param ustar_d  The USTAR.
 * @param u10  The U10 constant.
 * @param width_band  The width of the bands.
 * @param height_band  The height of the bands.
 */
__global__ void ustar_kernel(float *zom_d, float *d0_d, float *ustar_d, float u10, int width_band, int height_band);

/**
 * @brief  Compute the KB-1 stability parameter of the bands.
 *
 * @param zom_d  The ZOM.
 * @param ustar_d  The USTAR.
 * @param pai_d  The PAI.
 * @param kb1_d  The KB-1.
 * @param ndvi_d  The NDVI.
 * @param width_band  The width of the bands.
 * @param height_band  The height of the bands.
 * @param ndvi_max  The NDVI max value.
 * @param ndvi_min  The NDVI min value.
 */
__global__ void kb_kernel(float *zom_d, float *ustar_d, float *pai_d, float *kb1_d, float *ndvi_d, int width_band, int height_band, float ndvi_max, float ndvi_min);

/**
 * @brief  Compute the aerodynamic resistance of the bands.
 *
 * @param zom_d  The ZOM.
 * @param d0_d  The D0.
 * @param ustar_d  The USTAR.
 * @param kb1_d  The KB-1.
 * @param rah_d  The RAH.
 * @param width_band  The width of the bands.
 * @param height_band  The height of the bands.
 */
__global__ void aerodynamic_resistance_kernel(float *zom_d, float *d0_d, float *ustar_d, float *kb1_d, float *rah_d, int width_band, int height_band);

/**
 * @brief  Compute the sensible heat flux of the bands.
 *
 * @param surface_temperature_d  The surface temperature.
 * @param rah_d  The RAH.
 * @param net_radiation_d  The net radiation.
 * @param soil_heat_d  The soil heat.
 * @param sensible_heat_flux_d  The sensible heat flux.
 * @param a  The A constant.
 * @param b  The B constant.
 * @param width_band  The width of the bands.
 * @param height_band  The height of the bands.
 */
__global__ void sensible_heat_flux_kernel(float *surface_temperature_d, float *rah_d, float *net_radiation_d, float *soil_heat_d, float *sensible_heat_flux_d, float a, float b, int width_band, int height_band);

/**
 * @brief  Compute the latent heat flux of the bands.
 *
 * @param net_radiation_d  The net radiation.
 * @param soil_heat_d  The soil heat.
 * @param sensible_heat_flux_d  The sensible heat flux.
 * @param latent_heat_flux_d  The latent heat flux.
 * @param width_band  The width of the bands.
 * @param height_band  The height of the bands.
 */
__global__ void latent_heat_flux_kernel(float *net_radiation_d, float *soil_heat_d, float *sensible_heat_flux_d, float *latent_heat_flux_d, int width_band, int height_band);

/**
 * @brief  Compute the net radiation 24h of the bands.
 *
 * @param albedo_d  The albedo.
 * @param Rs24h  The Rs24h.
 * @param Ra24h  The Ra24h.
 * @param net_radiation_24h_d  The net radiation 24h.
 * @param width_band  The width of the bands.
 * @param height_band  The height of the bands.
 */
__global__ void net_radiation_24h_kernel(float *albedo_d, float Rs24h, float Ra24h, float *net_radiation_24h_d, int width_band, int height_band);

/**
 * @brief  Compute the evapotranspiration fraction of the bands.
 *
 * @param net_radiation_d  The net radiation.
 * @param soil_heat_d  The soil heat.
 * @param latent_heat_flux_d  The latent heat flux.
 * @param evapotranspiration_fraction_d  The evapotranspiration fraction.
 * @param width_band  The width of the bands.
 * @param height_band  The height of the bands.
 */
__global__ void evapotranspiration_fraction_kernel(float *net_radiation_d, float *soil_heat_d, float *latent_heat_flux_d, float *evapotranspiration_fraction_d, int width_band, int height_band);

/**
 * @brief  Compute the sensible heat flux 24h of the bands.
 *
 * @param net_radiation_24h_d  The net radiation 24h.
 * @param evapotranspiration_fraction_d  The evapotranspiration fraction.
 * @param sensible_heat_flux_24h_d  The sensible heat flux 24h.
 * @param width_band  The width of the bands.
 * @param height_band  The height of the bands.
 */
__global__ void sensible_heat_flux_24h_kernel(float *net_radiation_24h_d, float *evapotranspiration_fraction_d, float *sensible_heat_flux_24h_d, int width_band, int height_band);

/**
 * @brief  Compute the latent heat flux 24h of the bands.
 *
 * @param net_radiation_24h_d  The net radiation 24h.
 * @param evapotranspiration_fraction_d  The evapotranspiration fraction.
 * @param latent_heat_flux_24h_d  The latent heat flux 24h.
 * @param width_band  The width of the bands.
 * @param height_band  The height of the bands.
 */
__global__ void latent_heat_flux_24h_kernel(float *net_radiation_24h_d, float *evapotranspiration_fraction_d, float *latent_heat_flux_24h_d, int width_band, int height_band);

/**
 * @brief  Compute the evapotranspiration 24h of the bands.
 *
 * @param latent_heat_flux_24h_d  The latent heat flux 24h.
 * @param evapotranspiration_24h_d  The evapotranspiration 24h.
 * @param v7_max  The V7 max value.
 * @param v7_min  The V7 min value.
 * @param width_band  The width of the bands.
 * @param height_band  The height of the bands.
 */
__global__ void evapotranspiration_24h_kernel(float *latent_heat_flux_24h_d, float *evapotranspiration_24h_d, float v7_max, float v7_min, int width_band, int height_band);

/**
 * @brief  Compute the evapotranspiration of the bands.
 *
 * @param net_radiation_24h_d  The net radiation 24h.
 * @param evapotranspiration_fraction_d  The evapotranspiration fraction.
 * @param evapotranspiration_d  The evapotranspiration.
 * @param width_band  The width of the bands.
 * @param height_band  The height of the bands.
 */
__global__ void evapotranspiration_kernel(float *net_radiation_24h_d, float *evapotranspiration_fraction_d, float *evapotranspiration_d, int width_band, int height_band);

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

