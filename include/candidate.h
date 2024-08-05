#pragma once

#include "utils.h"

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
  int negative_neighbour;
  float coefficient_variation;
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

/**
 * @brief  Compares two Candidates based upon their position.
 * @param  a: First candidate.
 * @param  b: Second candidate.
 * @retval TRUE if they are at the same position, FALSE otherwise.
 */
bool equals(Candidate a, Candidate b);

/**
 * @brief  Compares two Candidates based upon their surface temperature.
 * @param  a: First candidate.
 * @param  b: Second candidate.
 * @retval TRUE if second candidate is greater than first one, and FALSE otherwise.
 */
bool compare_candidate_temperature(Candidate a, Candidate b);
