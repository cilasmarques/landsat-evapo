#pragma once

#include "utils.h"
#include "candidate.h"
#include "constants.h"
#include "parameters.h"

/**
 * @brief  Struct to manage the products calculation.
 */
struct Products
{
  int threads_num = 1;
  uint32_t width_band;
  uint32_t height_band;
  int nBytes_band;

  float H_pf_terra;
  float H_pq_terra;
  float rah_ini_pq_terra;
  float rah_ini_pf_terra;

  float *band1;
  float *band2;
  float *band3;
  float *band4;
  float *band5;
  float *band6;
  float *band7;
  float *band8;
  float *tal;

  float *radiance1;
  float *radiance2;
  float *radiance3;
  float *radiance4;
  float *radiance5;
  float *radiance6;
  float *radiance7;
  float *radiance8;

  float *reflectance1;
  float *reflectance2;
  float *reflectance3;
  float *reflectance4;
  float *reflectance5;
  float *reflectance6;
  float *reflectance7;
  float *reflectance8;

  float *albedo;
  float *ndvi;
  float *soil_heat;
  float *surface_temperature;
  float *net_radiation;

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

  float *devZom, *devTS, *devUstarR, *devUstarW, *devRahR, *devRahW, *devD0, *devKB1, *devH;

  float *band1_d, *band2_d, *band3_d, *band4_d, *band5_d, *band6_d, *band7_d, *band8_d;
  float *radiance1_d, *radiance2_d, *radiance3_d, *radiance4_d, *radiance5_d, *radiance6_d, *radiance7_d, *radiance8_d;
  float *reflectance1_d, *reflectance2_d, *reflectance3_d, *reflectance4_d, *reflectance5_d, *reflectance6_d, *reflectance7_d, *reflectance8_d;

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
  void radiance_function(MTL mtl, Sensor sensor);

  /**
   * @brief  The spectral reflectance for each band is computed.
   * @param  mtl: MTL struct.
   * @param  sensor: Sensor struct.
   */
  void reflectance_function(MTL mtl, Sensor sensor);

  /**
   * @brief  The surface albedo is computed.
   * @param  sensor: Sensor struct.
   */
  void albedo_function(MTL mtl, Sensor sensor);

  /**
   * @brief  The NDVI is computed.
   */
  void ndvi_function();

  /**
   * @brief  The PAI is computed.
   */
  void pai_function();

  /**
   * @brief  The LAI is computed.
   */
  void lai_function();

  /**
   * @brief  The EVI is computed.
   */
  void evi_function();

  /**
   * @brief  The emissivity is computed.
   */
  void enb_emissivity_function();

  /**
   * @brief  The emissivity is computed.
   */
  void eo_emissivity_function();

  /**
   * @brief  The emissivity is computed.
   */
  void ea_emissivity_function();

  /**
   * @brief  The surface temperature is computed.
   */
  void surface_temperature_function(MTL mtl);

  /**
   * @brief  The short wave radiation is computed.
   * @param  mtl: MTL struct.
   */
  void short_wave_radiation_function(MTL mtl);

  /**
   * @brief  The large wave radiation is computed.
   */
  void large_wave_radiation_surface_function();

  /**
   * @brief  The large wave radiation is computed.
   * @param  temperature: Pixel's temperature.
   */
  void large_wave_radiation_atmosphere_function(double temperature);

  /**
   * @brief  The net radiation is computed.
   */
  void net_radiation_function();

  /**
   * @brief  The soil heat flux is computed.
   */
  void soil_heat_flux_function();

  /**
   * @brief  The d0 is computed.
   */
  void d0_fuction();

  /**
   * @brief  The kb is computed.
   * @param  ndvi_max: Maximum NDVI.
   * @param  ndvi_min: Minimum NDVI.
   */
  void kb_function(double ndvi_max, double ndvi_min);

  /**
   * @brief  The zom is computed.
   * @param  A_ZOM: Coefficient A.
   * @param  B_ZOM: Coefficient B.
   */
  void zom_fuction(double A_ZOM, double B_ZOM);

  /**
   * @brief  The ustar is computed.
   * @param  u10: Wind speed at 10 m.
   */
  void ustar_fuction(double u10);

  /**
   * @brief  The aerodynamic resistance is computed.
   */
  void aerodynamic_resistance_fuction();

  /**
   * @brief  The sensible heat flux is computed.
   * @param  a: Coefficient A.
   * @param  b: Coefficient B.
   */
  void sensible_heat_flux_function(double a, double b);

  /**
   * @brief  The latent heat flux is computed.
   */
  void latent_heat_flux_function();

  /**
   * @brief  The net radiation is computed.
   * @param  Ra24h: Net radiation 24h.
   * @param  Rs24h: Solar radiation 24h.
   */
  void net_radiation_24h_function(double Ra24h, double Rs24h);

  /**
   * @brief  The evapotranspiration fraction is computed.
   */
  void evapotranspiration_fraction_fuction();

  /**
   * @brief  The sensible heat flux is computed.
   */
  void sensible_heat_flux_24h_fuction();

  /**
   * @brief  The latent heat flux is computed.
   */
  void latent_heat_flux_24h_function();

  /**
   * @brief  The evapotranspiration is computed.
   */
  void evapotranspiration_24h_function(Station station);

  /**
   * @brief  The evapotranspiration is computed.
   */
  void evapotranspiration_function();

  /**
   * @brief  The  aerodynamic resistance convergence is computed.
   * @param  ndvi_min: Minimum NDVI.
   * @param  ndvi_max: Maximum NDVI.
   * @param  hot_pixel: Hot pixel.
   * @param  cold_pixel: Cold pixel.
   * @return  string: Time message.
   */
  string rah_correction_function_serial(double ndvi_min, double ndvi_max, Candidate hot_pixel, Candidate cold_pixel);

  /**
   * @brief  The  aerodynamic resistance convergence is computed.
   * @param  ndvi_min: Minimum NDVI.
   * @param  ndvi_max: Maximum NDVI.
   * @param  hot_pixel: Hot pixel.
   * @param  cold_pixel: Cold pixel.
   * @return  string: Time message.
   */
  string rah_correction_function_threads(double ndvi_min, double ndvi_max, Candidate hot_pixel, Candidate cold_pixel);

  /**
   * @brief  The  aerodynamic resistance convergence is computed.
   * @param  ndvi_min: Minimum NDVI.
   * @param  ndvi_max: Maximum NDVI.
   * @param  hot_pixel: Hot pixel.
   * @param  cold_pixel: Cold pixel.
   * @return  string: Time message.
   */
  string rah_correction_function_blocks(double ndvi_min, double ndvi_max, Candidate hot_pixel, Candidate cold_pixel);
};
