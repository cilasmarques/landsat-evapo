#pragma once

#include "utils.h"
#include "candidate.h"
#include "constants.h"
#include "parameters.h"

#ifdef __CUDACC__
#include "tensor.cuh"
#endif

/**
 * @brief  Struct to manage the products calculation.
 */
struct Products
{
  int threads_num = 1;
  int blocks_num;

  int nBytes_band;
  uint32_t width_band;
  uint32_t height_band;

  float H_pf_terra;
  float H_pq_terra;
  float rah_ini_pq_terra;
  float rah_ini_pf_terra;

  float *band_blue;
  float *band_green;
  float *band_red;
  float *band_nir;
  float *band_swir1;
  float *band_termal;
  float *band_swir2;
  float *tal;

  float *radiance_blue;
  float *radiance_green;
  float *radiance_red;
  float *radiance_nir;
  float *radiance_swir1;
  float *radiance_termal;
  float *radiance_swir2;

  float *reflectance_blue;
  float *reflectance_green;
  float *reflectance_red;
  float *reflectance_nir;
  float *reflectance_swir1;
  float *reflectance_termal;
  float *reflectance_swir2;

  float *albedo;
  float *ndvi;
  float *soil_heat;
  float *surface_temperature;
  float *net_radiation;

  float *savi;
  float *lai;
  float *evi;
  float *pai;
  float *enb_emissivity;
  float *eo_emissivity;
  float *ea_emissivity;
  float *short_wave_radiation;
  float *large_wave_radiation_surface;
  float *large_wave_radiation_atmosphere;

  float *d0;
  float *zom;
  float *ustar;
  float *kb1;
  float *aerodynamic_resistance;
  float *sensible_heat_flux;

  float *latent_heat_flux;
  float *net_radiation_24h;
  float *evapotranspiration_fraction;
  float *sensible_heat_flux_24h;
  float *latent_heat_flux_24h;
  float *evapotranspiration_24h;
  float *evapotranspiration;

  float *band_blue_d, *band_green_d, *band_red_d, *band_nir_d, *band_swir1_d, *band_termal_d, *band_swir2_d;
  float *radiance_blue_d, *radiance_green_d, *radiance_red_d, *radiance_nir_d, *radiance_swir1_d, *radiance_termal_d, *radiance_swir2_d;
  float *reflectance_blue_d, *reflectance_green_d, *reflectance_red_d, *reflectance_nir_d, *reflectance_swir1_d, *reflectance_termal_d, *reflectance_swir2_d;

  float *tal_d, *albedo_d, *ndvi_d, *pai_d, *savi_d, *lai_d, *evi_d;
  float *enb_d, *eo_d, *ea_d, *short_wave_radiation_d, *large_wave_radiation_surface_d, *large_wave_radiation_atmosphere_d;
  float *soil_heat_d, *surface_temperature_d, *net_radiation_d, *d0_d, *kb1_d, *zom_d, *ustar_d, *rah_d, *sensible_heat_flux_d;
  float *latent_heat_flux_d, *net_radiation_24h_d, *evapotranspiration_fraction_d, *sensible_heat_flux_24h_d, *latent_heat_flux_24h_d, *evapotranspiration_24h_d, *evapotranspiration_d;

  // === Used in tensor implementation ===
  #ifdef __CUDACC__
    Tensor tensors;
    float *only1, *only1_d;
    float *tensor_aux1_d, *tensor_aux2_d;
    float *Re_star_d;
    float *Ct_star_d;
    float *beta_d;
    float *nec_terra_d;
    float *kb1_fst_part_d;
    float *kb1_sec_part_d;
    float *kb1s_d;
    float *fc_d;
    float *fs_d;
    float *fspow_d;
    float *fcpow_d;
  #endif
  // === Used in tensor implementation ===

  /**
   * @brief  Constructor.
   */
  Products();

  /**
   * @brief  Constructor.
   * @param  width_band: Band width.
   * @param  height_band: Band height.
   */
  Products(uint32_t width_band, uint32_t height_band, int threads_num);

  /**
   * @brief  Destructor.
   */
  void close();

  /**
   * @brief  The spectral radiance for each band is computed.
   * @param  mtl: MTL struct.
   * @param  sensor: Sensor struct.
   */
  string radiance_function(MTL mtl);

  /**
   * @brief  The spectral reflectance for each band is computed.
   * @param  mtl: MTL struct.
   * @param  sensor: Sensor struct.
   */
  string reflectance_function(MTL mtl);

  string invalid_rad_ref_function();

  /**
   * @brief  The surface albedo is computed.
   * @param  sensor: Sensor struct.
   */
  string albedo_function(MTL mtl);

  /**
   * @brief  The NDVI is computed.
   */
  string ndvi_function();

  /**
   * @brief  The PAI is computed.
   */
  string pai_function();

  /**
   * @brief  The LAI is computed.
   */
  string lai_function();

  /**
   * @brief  The EVI is computed.
   */
  string evi_function();

  /**
   * @brief  The emissivity is computed.
   */
  string enb_emissivity_function();

  /**
   * @brief  The emissivity is computed.
   */
  string eo_emissivity_function();

  /**
   * @brief  The emissivity is computed.
   */
  string ea_emissivity_function();

  /**
   * @brief  The surface temperature is computed.
   */
  string surface_temperature_function(MTL mtl);

  /**
   * @brief  The short wave radiation is computed.
   * @param  mtl: MTL struct.
   */
  string short_wave_radiation_function(MTL mtl);

  /**
   * @brief  The large wave radiation is computed.
   */
  string large_wave_radiation_surface_function();

  /**
   * @brief  The large wave radiation is computed.
   * @param  temperature: Pixel's temperature.
   */
  string large_wave_radiation_atmosphere_function(float temperature);

  /**
   * @brief  The net radiation is computed.
   */
  string net_radiation_function();

  /**
   * @brief  The soil heat flux is computed.
   */
  string soil_heat_flux_function();

  /**
   * @brief  The d0 is computed.
   */
  string d0_fuction();

  /**
   * @brief  The kb is computed.
   * @param  ndvi_max: Maximum NDVI.
   * @param  ndvi_min: Minimum NDVI.
   */
  string kb_function(float ndvi_max, float ndvi_min);

  /**
   * @brief  The zom is computed.
   * @param  A_ZOM: Coefficient A.
   * @param  B_ZOM: Coefficient B.
   */
  string zom_fuction(float A_ZOM, float B_ZOM);

  /**
   * @brief  The ustar is computed.
   * @param  u10: Wind speed at 10 m.
   */
  string ustar_fuction(float u10);

  /**
   * @brief  The aerodynamic resistance is computed.
   */
  string aerodynamic_resistance_fuction();

  /**
   * @brief  The sensible heat flux is computed.
   * @param  a: Coefficient A.
   * @param  b: Coefficient B.
   */
  string sensible_heat_flux_function(float a, float b);

  /**
   * @brief  The latent heat flux is computed.
   */
  string latent_heat_flux_function();

  /**
   * @brief  The net radiation is computed.
   * @param  Ra24h: Net radiation 24h.
   * @param  Rs24h: Solar radiation 24h.
   */
  string net_radiation_24h_function(float Ra24h, float Rs24h);

  /**
   * @brief  The evapotranspiration fraction is computed.
   */
  string evapotranspiration_fraction_fuction();

  /**
   * @brief  The sensible heat flux is computed.
   */
  string sensible_heat_flux_24h_fuction();

  /**
   * @brief  The latent heat flux is computed.
   */
  string latent_heat_flux_24h_function();

  /**
   * @brief  The evapotranspiration is computed.
   */
  string evapotranspiration_24h_function(Station station);

  /**
   * @brief  The evapotranspiration is computed.
   */
  string evapotranspiration_function();

  /**
   * @brief  The  aerodynamic resistance convergence is computed.
   * @param  ndvi_min: Minimum NDVI.
   * @param  ndvi_max: Maximum NDVI.
   * @param  hot_pixel: Hot pixel.
   * @param  cold_pixel: Cold pixel.
   * @return  string: Time message.
   */
  string rah_correction_function_serial(float ndvi_min, float ndvi_max, Candidate hot_pixel, Candidate cold_pixel);

  /**
   * @brief  The  aerodynamic resistance convergence is computed.
   * @param  ndvi_min: Minimum NDVI.
   * @param  ndvi_max: Maximum NDVI.
   * @param  hot_pixel: Hot pixel.
   * @param  cold_pixel: Cold pixel.
   * @return  string: Time message.
   */
  string rah_correction_function_threads(float ndvi_min, float ndvi_max, Candidate hot_pixel, Candidate cold_pixel);

  /**
   * @brief  The  aerodynamic resistance convergence is computed.
   * @param  ndvi_min: Minimum NDVI.
   * @param  ndvi_max: Maximum NDVI.
   * @param  hot_pixel: Hot pixel.
   * @param  cold_pixel: Cold pixel.
   * @return  string: Time message.
   */
  string rah_correction_function_blocks(float ndvi_min, float ndvi_max, Candidate hot_pixel, Candidate cold_pixel);
};
