#pragma once

#include "sensors.cuh"

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

/**
 * @brief  Struct representing a hot or cold pixel candidate.
 */
struct Candidate
{
  int line, col;
  float aerodynamic_resistance;
  float ndvi, temperature, ustar;
  float net_radiation, soil_heat_flux, ho, zom;

  /**
   * @brief  Empty constructor, all attributes are initialized with 0.
   */
  CUDA_HOSTDEV Candidate();

  /**
   * @brief  Constructor with initialization values to attributes.
   * @param  ndvi: Pixel's NDVI.
   * @param  temperature: Pixel's surface temperature.
   * @param  net_radiation: Pixel's net radiation.
   * @param  soil_heat_flux: Pixel's soil heat flux.
   * @param  ho: Pixel's ho.
   * @param  line: Pixel's line on TIFF.
   * @param  col: Pixel's column on TIFF.
   */
  CUDA_HOSTDEV Candidate(float ndvi, float temperature, float net_radiation, float soil_heat_flux, float ho, int line, int col);

  /**
   * @brief  Update Pixel's aerodynamic resistance for a new value.
   * @param  newRah: new value of aerodynamic resistance.
   */
  void setAerodynamicResistance(float newRah);
};

struct CompareCandidateTemperature
{
  __host__ __device__ bool operator()(Candidate a, Candidate b)
  {
    // Assuming Candidate has a member variable 'temperature'
    bool result = a.temperature < b.temperature;

    if (a.temperature == b.temperature)
      result = a.ndvi < b.ndvi;

    return result;
  }
};

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

  int band_size;
  int band_bytes;

  int *stop_condition, *stop_condition_d;
  Candidate *d_hotCandidates, *d_coldCandidates;

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
  string read_data(TIFF **landsat_bands);

  /**
   * @brief Compute the initial products.
   *
   * @param  station: Station struct.
   * @return string with the time spent.
   */
  string compute_Rn_G(Products products, Station station, MTL mtl);

  /**
   * @brief Select the cold and hot endmembers
   *
   * @param  method: Method to select the endmembers.
   * @return string with the time spent.
   */
  string select_endmembers(int method);

  /**
   * @brief make the rah cycle converge
   *
   * @param  station: Station struct.
   * @param  method: Method to converge the rah cycle.
   *
   * @return string with the time spent.
   */
  string converge_rah_cycle(Products products, Station station, int method);

  /**
   * @brief Compute the final products.
   *
   * @param  station: Station struct.
   * @return string with the time spent.
   */
  string compute_H_ET(Products products, Station station, MTL mtl);

  /**
   * @brief Copy the products to the host.
   *
   * @return string with the time spent.
   */
  string host_products();

  /**
   * @brief Copy the products to the host.
   *
   * @return string with the time spent.
   */
  void device_products();

  /**
   * @brief Save the products.
   *
   * @param  output_path: Path to save the products.
   * @return string with the time spent.
   */
  string save_products(string output_path);

  /**
   * @brief Print the products.
   *
   * @param  output_path: Path to save the products.
   * @return string with the time spent.
   */
  string print_products(string output_path);

  /**
   * @brief Close the TIFF files.
   */
  void close(TIFF **landsat_bands);
};