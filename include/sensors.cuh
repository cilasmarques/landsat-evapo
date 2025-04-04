#pragma once

#include "constants.h"

/**
 * @brief  Struct to hold some metadata informations.
 */
struct MTL {
    float image_hour;
    int number_sensor, julian_day, year;
    float sun_elevation, distance_earth_sun;

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
struct Station {
    vector<vector<string>> info;
    float temperature_image;
    float v6, v7_max, v7_min;
    float latitude, longitude;

    const int WIND_SPEED = 3;
    const float SURFACE_ROUGHNESS = 0.024;
    const float A_ZOM = -3;
    const float B_ZOM = 6.47;
    const float INTERNALIZATION_FACTOR = 0.16;

    /**
     * @brief  Empty constructor. Set temperature_image to 0.
     */
    Station();

    /**
     * @brief  Constructor.
     * @param  station_data_path: Weather station data file.
     * @param  image_hour: Image hour.
     */
    Station(string station_data_path, float image_hour);
};

/**
 * @brief  Struct to manage the products calculation.
 */
struct Landsat {
    TIFF *landsat_bands[9];
    uint16_t sample_bands;
    uint32_t height_band;
    uint32_t width_band;

    /**
     * @brief  Empty constructor. Set all attributes to 0.
     */
    Landsat();

    /**
     * @brief  Constructor.
     * @param  bands_paths: Paths to the bands.
     */
    Landsat(string bands_paths[]);
};

/**
 * @brief  Prints a pointer.
 *
 * @param pointer: Pointer to be printed.
 * @param height: Height of the pointer.
 * @param width: Width of the pointer.
 */
void printLinearPointer(float *pointer, int height, int width);

/**
 * @brief  Saves a TIFF file.
 *
 * @param path: Path to save the TIFF file.
 * @param data: Data to be saved.
 * @param height: Height of the data.
 * @param width: Width of the data.
 */
void saveTiff(string path, float *data, int height, int width);
