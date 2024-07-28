#pragma once

#include "constants.h"

/**
 * @brief  Struct to hold some metadata informations.
 */
struct MTL
{
  double image_hour;
  int number_sensor, julian_day, year;
  double sun_elevation, distance_earth_sun;

  float *rad_mult;
  float *rad_add;
  float *ref_mult;
  float *ref_add;
  float *ref_w_coeff;

  float *rad_mult_d;
  float *rad_add_d;
  float *ref_mult_d;
  float *ref_add_d;
  float *ref_w_coeff_d;

  /**
   * @brief  Empty constructor. Setting all attributes to 0.
   */
  MTL();

  /**
   * @brief  Constructor receiving the metadata path.
   * @param  metadata_path: Metadata file path.
   */
  MTL(string metadata_path);
};


/**
 * @brief  Struct to hold the weather station data.
 */
struct Station
{
  vector<vector<string>> info;
  double temperature_image;
  double v6, v7_max, v7_min;
  double latitude, longitude;

  const int WIND_SPEED = 3;
  const double SURFACE_ROUGHNESS = 0.024;
  const double A_ZOM = -3;
  const double B_ZOM = 6.47;
  const double INTERNALIZATION_FACTOR = 0.16;

  /**
   * @brief  Empty constructor. Set temperature_image to 0.
   */
  Station();

  /**
   * @brief  Constructor.
   * @param  station_data_path: Weather station data file.
   * @param  image_hour: Image hour.
   */
  Station(string station_data_path, double image_hour);
};