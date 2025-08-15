#pragma once

#include "constants.h"

/**
 * @brief  Struct to hold some metadata informations.
 */
struct MTL {
    float image_hour;
    int number_sensor, julian_day, year;
    float sun_elevation, distance_earth_sun;

    double *rad_mult;
    double *rad_add;
    double *ref_mult;
    double *ref_add;
    double *ref_w_coeff;

    double *rad_mult_d;
    double *rad_add_d;
    double *ref_mult_d;
    double *ref_add_d;
    double *ref_w_coeff_d;

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
void printLinearPointer(double *pointer, int height, int width);

/**
 * @brief  Saves a TIFF file.
 *
 * @param path: Path to save the TIFF file.
 * @param data: Data to be saved.
 * @param height: Height of the data.
 * @param width: Width of the data.
 */
void saveTiff(string path, double *data, int height, int width);
