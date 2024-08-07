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

  Candidate hot_pixel;
  Candidate cold_pixel;

  MTL mtl;
  Products products;

  /**
   * @brief  Constructor.
   * @param  bands_paths: Paths to the bands.
   * @param  tal_path: Path to the TAL file.
   * @param  land_cover_path: Path to the land cover file.
   */
  Landsat(string bands_paths[], MTL mtl, int threads_num);

  string select_endmembers(int method);

  string converge_rah_cycle(Station station, int method);

  string compute_Rn_G(Station station);

  string compute_H_ET(Station station);

  string save_products(string output_path);

  void close();
};