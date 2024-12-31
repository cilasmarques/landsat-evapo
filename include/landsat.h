#pragma once

#include "utils.h"
#include "products.h"
#include "constants.h"
#include "candidate.h"
#include "constants.h"
#include "endmembers.h"
#include "parameters.h"

/**
 * @brief  Struct to manage the products calculation.
 */
struct Landsat
{
  TIFF *bands_resampled[9];
  uint16_t sample_bands;
  uint32_t height_band;
  uint32_t width_band;
  int threads_num;

  Candidate *d_hotCandidates, *d_coldCandidates;

  MTL mtl;
  Products products;

  /**
   * @brief  Constructor.
   * @param  bands_paths: Paths to the bands.
   * @param  tal_path: Path to the TAL file.
   * @param  land_cover_path: Path to the land cover file.
   */
  Landsat(string bands_paths[], MTL mtl, int threads_num);

  /**
   * @brief  Destructor.
   */
  void close();

  /**
   * @brief Compute the initial products.
   * 
   * @param  station: Station struct.
   * @return string with the time spent.
   */
  string compute_Rn_G(Station station);

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
  string converge_rah_cycle(Station station, int method);

  /**
   * @brief Compute the final products.
   * 
   * @param  station: Station struct.
   * @return string with the time spent.
   */
  string compute_H_ET(Station station);

  /**
   * @brief Copy the products to the host.
   * 
   * @return string with the time spent.
   */
  string copy_to_host();

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
};