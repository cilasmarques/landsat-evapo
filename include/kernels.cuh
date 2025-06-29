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

/**
 * @brief  Compute the radiance of the bands.
 *
 * @param band_d  The band.
 * @param radiance_d  The radiance.
 * @param rad_add_d  The radiance add value.
 * @param rad_mult_d  The radiance mult value.
 * @param band_idx  The band index.
 */
__global__ void rad_kernel(float *band_d, float *radiance_d, float *rad_add_d, float *rad_mult_d, int band_idx);

/**
 * @brief  Compute the reflectance of the bands.
 *
 * @param band_d  The band.
 * @param reflectance_d  The reflectance.
 * @param ref_add_d  The reflectance add value.
 * @param ref_mult_d  The reflectance mult value.
 * @param sin_sun  The sin of the sun.
 * @param band_idx  The band index.
 */
__global__ void ref_kernel(float *band_d, float *reflectance_d, float *ref_add_d, float *ref_mult_d, float sin_sun, int band_idx);

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
 */
__global__ void albedo_kernel(float *reflectance_blue_d, float *reflectance_green_d, float *reflectance_red_d, float *reflectance_nir_d, float *reflectance_swir1_d, float *reflectance_swir2_d, float *tal_d, float *albedo_d, float *ref_w_coeff_d);

/**
 * @brief  Compute the NDVI of the bands.
 *
 * @param reflectance_nir_d  The NIR band.
 * @param reflectance_red_d  The red band.
 * @param ndvi_d  The NDVI.
 */
__global__ void ndvi_kernel(float *reflectance_nir_d, float *reflectance_red_d, float *ndvi_d);

/**
 * @brief  Compute the TAL of the bands.
 *
 * @param reflectance_nir_d  The NIR reflectance.
 * @param reflectance_red_d  The red reflectance.
 * @param pai_d The PAI.
 */
__global__ void pai_kernel(float *reflectance_nir_d, float *reflectance_red_d, float *pai_d);

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
 * @param enb_d  The ENB.
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
 * @brief  Compute the EA of the bands.
 *
 * @param tal_d  The TAL.
 * @param ea_d  The EA.
 */
__global__ void ea_kernel(float *tal_d, float *ea_d);

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
 * @brief  Compute the short wave radiation of the bands.
 *
 * @param tal_d  The TAL.
 * @param short_wave_radiation_d  The short wave radiation.
 * @param sun_elevation  The sun elevation.
 * @param distance_earth_sun  The distance between the earth and the sun.
 * @param pi  The pi constant.
 */
__global__ void short_wave_radiation_kernel(float *tal_d, float *short_wave_radiation_d, float sun_elevation, float distance_earth_sun, float pi);

/**
 * @brief  Compute the large wave radiation of the bands.
 *
 * @param surface_temperature_d  The surface temperature.
 * @param eo_d  The EO.
 * @param ea_d  The EA.
 * @param large_wave_radiation_atmosphere_d  The large wave radiation atmosphere.
 * @param large_wave_radiation_surface_d  The large wave radiation surface.
 * @param temperature  The surface temperature.
 */
__global__ void large_waves_radiation_kernel(float *surface_temperature_d, float *eo_d, float *ea_d, float *large_wave_radiation_atmosphere_d, float *large_wave_radiation_surface_d, float temperature);

/**
 * @brief  Compute the net radiation of the bands.
 *
 * @param short_wave_radiation_d  The short wave radiation.
 * @param albedo_d  The albedo.
 * @param large_wave_radiation_atmosphere_d  The large wave radiation atmosphere.
 * @param large_wave_radiation_surface_d  The large wave radiation surface.
 * @param eo_d  The EO.
 * @param net_radiation_d  The net radiation.
 */
__global__ void net_radiation_kernel(float *short_wave_radiation_d, float *albedo_d, float *large_wave_radiation_atmosphere_d, float *large_wave_radiation_surface_d, float *eo_d, float *net_radiation_d);

/**
 * @brief  Compute the soil heat of the bands.
 *
 * @param ndvi_d  The NDVI.
 * @param albedo_d  The albedo.
 * @param surface_temperature_d  The surface temperature.
 * @param net_radiation_d  The net radiation.
 * @param soil_heat_d  The soil heat.
 */
__global__ void soil_heat_kernel(float *ndvi_d, float *albedo_d, float *surface_temperature_d, float *net_radiation_d, float *soil_heat_d);

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
 * @param hotCandidates_d The hot candidates array.
 * @param coldCandidates_d The cold candidates array.
 * @param indexes_d The indexes of the hot and cold arrays.
 * @param ndvi_d The NDVI array.
 * @param surface_temperature_d The surface temperature array.
 * @param albedo_d The albedo array.
 * @param net_radiation_d The net radiation array.
 * @param soil_heat_d The soil heat array.
 * @param ndviQuartileLow The NDVI low quartile.
 * @param ndviQuartileHigh The NDVI high quartile.
 * @param tsQuartileLow The surface temperature low quartile.
 * @param tsQuartileMid The surface temperature mid quartile.
 * @param tsQuartileHigh The surface temperature high quartile.
 * @param albedoQuartileLow The albedo low quartile.
 * @param albedoQuartileMid The albedo mid quartile.
 * @param albedoQuartileHigh The albedo high quartile.
 */
__global__ void process_pixels_STEEP(Endmember *hotCandidates_d, Endmember *coldCandidates_d, int *indexes_d, float *ndvi_d, float *surface_temperature_d, float *albedo_d, float *net_radiation_d, float *soil_heat_d, float ndviQuartileLow, float ndviQuartileHigh, float tsQuartileLow, float tsQuartileMid, float tsQuartileHigh, float albedoQuartileLow, float albedoQuartileMid, float albedoQuartileHigh);

/**
 * @brief Process the pixels of the target arrays and store the candidates in the hot and cold arrays.
 *
 * @param hotCandidates_d The hot candidates array.
 * @param coldCandidates_d The cold candidates array.
 * @param indexes_d The indexes of the hot and cold arrays.
 * @param ndvi_d The NDVI array.
 * @param surface_temperature_d The surface temperature array.
 * @param albedo_d The albedo array.
 * @param net_radiation_d The net radiation array.
 * @param soil_heat_d The soil heat array.
 * @param ndviHOTQuartile The NDVI hot quartile.
 * @param ndviCOLDQuartile The NDVI cold quartile.
 * @param tsHOTQuartile The surface temperature hot quartile.
 * @param tsCOLDQuartile The surface temperature cold quartile.
 * @param albedoHOTQuartile The albedo hot quartile.
 * @param albedoCOLDQuartile The albedo cold quartile.
 */
__global__ void process_pixels_ASEBAL(Endmember *hotCandidates_d, Endmember *coldCandidates_d, int *indexes_d, float *ndvi_d, float *surface_temperature_d, float *albedo_d, float *net_radiation_d, float *soil_heat_d, float ndviHOTQuartile, float ndviCOLDQuartile, float tsHOTQuartile, float tsCOLDQuartile, float albedoHOTQuartile, float albedoCOLDQuartile);

/**
 * @brief  Compute the zero plane displacement height_d of the bands.
 *
 * @param pai_d  The PAI.
 * @param d0_d  The D0.
 * @param CD1  The CD1 constant.
 * @param HGHT  The HGHT constant.
 */
__global__ void d0_kernel(float *pai_d, float *d0_d, float CD1, float HGHT);

/**
 * @brief  Compute the roughness length for momentum of the bands.
 *
 * @param d0_d  The D0.
 * @param pai_d  The PAI.
 * @param zom_d  The ZOM.
 * @param A_ZOM  The A_ZOM constant.
 * @param B_ZOM  The B_ZOM constant.
 */
__global__ void zom_kernel_STEEP(float *d0_d, float *pai_d, float *zom_d, float A_ZOM, float B_ZOM);

/**
 * @brief  Compute the rah of the bands.
 *
 * @param ndvi_d  The NDVI.
 * @param albedo_d  The albedo.
 * @param zom_d  The ZOM.
 * @param A_ZOM  The A_ZOM constant.
 * @param B_ZOM  The B_ZOM constant.
 */
__global__ void zom_kernel_ASEBAL(float *ndvi_d, float *albedo_d,  float *zom_d, float A_ZOM, float B_ZOM);

/**
 * @brief  Compute the ustar of the bands.
 *
 * @param zom_d  The ZOM.
 * @param d0_d  The D0.
 * @param ustar_d  The USTAR.
 * @param u10  The U10 constant.
 */
__global__ void ustar_kernel_STEEP(float *zom_d, float *d0_d, float *ustar_d, float u10);

/**
 * @brief  Compute the ustar of the bands.
 *
 * @param zom_d  The ZOM.
 * @param ustar_d  The USTAR.
 * @param u200  The U200 constant.
 */
__global__ void ustar_kernel_ASEBAL(float *zom_d, float *ustar_d, float u200);

/**
 * @brief  Compute the KB-1 stability parameter of the bands.
 *
 * @param zom_d  The ZOM.
 * @param ustar_d  The USTAR.
 * @param pai_d  The PAI.
 * @param kb1_d  The KB-1.
 * @param ndvi_d  The NDVI.
 * @param ndvi_max  The NDVI max value.
 * @param ndvi_min  The NDVI min value.
 */
__global__ void kb_kernel(float *zom_d, float *ustar_d, float *pai_d, float *kb1_d, float *ndvi_d, float ndvi_max, float ndvi_min);

/**
 * @brief  Compute the aerodynamic resistance of the bands.
 *
 * @param zom_d  The ZOM.
 * @param d0_d  The D0.
 * @param ustar_d  The USTAR.
 * @param kb1_d  The KB-1.
 * @param rah_d  The RAH.
 */
__global__ void aerodynamic_resistance_kernel_STEEP(float *zom_d, float *d0_d, float *ustar_d, float *kb1_d, float *rah_d);

/**
 * @brief  Compute the aerodynamic resistance of the bands.
 *
 * @param ustar_d  The USTAR.
 * @param rah_d  The RAH.
 */
__global__ void aerodynamic_resistance_kernel_ASEBAL(float *ustar_d, float *rah_d);

/**
 * @brief  Compute the rah correction cycle. (STEEP algorithm)
 *
 * @param net_radiation_d  Net radiation
 * @param soil_heat_flux_d  Soil heat flux
 * @param ndvi_d  NDVI
 * @param surface_temperature_d  Surface temperature
 * @param d0_d  Zero plane displacement height_d
 * @param kb1_d  KB-1 stability parameter
 * @param zom_d  Roughness length for momentum
 * @param ustar_d  Ustar pointer
 * @param rah_d  Rah pointer
 * @param H_d  Sensible heat flux
 * @param ndvi_max  NDVI max value
 * @param ndvi_min  NDVI min value
 */
__global__ void rah_correction_cycle_STEEP(float *net_radiation_d, float *soil_heat_flux_d, float *ndvi_d, float *surface_temperature_d, float *d0_d, float *kb1_d, float *zom_d, float *ustar_d, float *rah_d, float *H_d, float ndvi_max, float ndvi_min);

/**
 * @brief  Compute the rah correction cycle. (STEEP algorithm)
 *
 * @param net_radiation_d  Net radiation
 * @param soil_heat_flux_d  Soil heat flux
 * @param ndvi_d  NDVI
 * @param surface_temperature_d  Surface temperature
 * @param kb1_d  KB-1 stability parameter
 * @param zom_d  Roughness length for momentum
 * @param ustar_d  Ustar pointer
 * @param rah_d  Rah pointer
 * @param H_d  Sensible heat flux
 * @param u200 U200
 * @param stop_condition Stop condition
 */
__global__ void rah_correction_cycle_ASEBAL(float *net_radiation_d, float *soil_heat_flux_d, float *ndvi_d, float *surface_temperature_d, float *kb1_d, float *zom_d, float *ustar_d, float *rah_d, float *H_d, float u200, int *stop_condition);

/**
 * @brief  Compute the sensible heat flux of the bands.
 *
 * @param surface_temperature_d  The surface temperature.
 * @param rah_d  The RAH.
 * @param net_radiation_d  The net radiation.
 * @param soil_heat_d  The soil heat.
 * @param sensible_heat_flux_d  The sensible heat flux.
 */
__global__ void sensible_heat_flux_kernel(float *surface_temperature_d, float *rah_d, float *net_radiation_d, float *soil_heat_d, float *sensible_heat_flux_d);

/**
 * @brief  Compute the latent heat flux of the bands.
 *
 * @param net_radiation_d  The net radiation.
 * @param soil_heat_d  The soil heat.
 * @param sensible_heat_flux_d  The sensible heat flux.
 * @param latent_heat_flux_d  The latent heat flux.
 */
__global__ void latent_heat_flux_kernel(float *net_radiation_d, float *soil_heat_d, float *sensible_heat_flux_d, float *latent_heat_flux_d);

/**
 * @brief  Compute the net radiation 24h of the bands.
 *
 * @param albedo_d  The albedo.
 * @param Rs24h  The Rs24h.
 * @param Ra24h  The Ra24h.
 * @param net_radiation_24h_d  The net radiation 24h.
 */
__global__ void net_radiation_24h_kernel(float *albedo_d, float Rs24h, float Ra24h, float *net_radiation_24h_d);

/**
 * @brief  Compute the evapotranspiration 24h of the bands.
 * 
 * @param surface_temperature_d  The surface temperature.
 * @param latent_heat_flux_d  The latent heat flux.
 * @param net_radiation_d  The net radiation.
 * @param soil_heat_d  The soil heat.
 * @param net_radiation_24h_d  The net radiation 24h.
 * @param evapotranspiration_24h_d  The evapotranspiration 24h.
 */
__global__ void evapotranspiration_24h_kernel(float *surface_temperature_d, float *latent_heat_flux_d, float *net_radiation_d, float *soil_heat_d, float *net_radiation_24h_d, float *evapotranspiration_24h_d);
