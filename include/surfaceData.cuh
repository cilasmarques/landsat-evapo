#pragma once

#include "sensors.cuh"

/**
 * @brief  Struct representing a hot or cold pixel candidate.
 */
struct Endmember {
    int line, col;
    float ndvi, temperature;

    /**
     * @brief  Empty constructor, all attributes are initialized with 0.
     */
    __host__ __device__ Endmember();

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
    __host__ __device__ Endmember(float ndvi, float temperature, int line, int col);
};

/**
 * @brief  Struct to compare two candidates by their NDVI and temperature.
 */
struct CompareEndmemberTemperature {
    __host__ __device__ bool operator()(Endmember a, Endmember b)
    {
        // Assuming Endmember has a member variable 'temperature'
        bool result = a.temperature < b.temperature;

        if (a.temperature == b.temperature)
            result = a.ndvi < b.ndvi;

        return result;
    }
};

/**
 * @brief  Struct to manage the products calculation.
 */
struct Products {
    uint32_t width_band;
    uint32_t height_band;

    int band_size;
    int band_bytes;

    int *stop_condition, *stop_condition_d;
    Endmember *hotCandidates_d, *coldCandidates_d;

    // Host pointers
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
    float *savi;
    float *lai;
    float *pai;

    float *soil_heat;
    float *net_radiation;
    float *surface_temperature;

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

    // Device pointers
    float *band_blue_d;
    float *band_green_d;
    float *band_red_d;
    float *band_nir_d;
    float *band_swir1_d;
    float *band_termal_d;
    float *band_swir2_d;
    float *tal_d;

    float *radiance_blue_d;
    float *radiance_green_d;
    float *radiance_red_d;
    float *radiance_nir_d;
    float *radiance_swir1_d;
    float *radiance_termal_d;
    float *radiance_swir2_d;

    float *reflectance_blue_d;
    float *reflectance_green_d;
    float *reflectance_red_d;
    float *reflectance_nir_d;
    float *reflectance_swir1_d;
    float *reflectance_termal_d;
    float *reflectance_swir2_d;

    float *albedo_d;
    float *ndvi_d;
    float *pai_d;
    float *savi_d;
    float *lai_d;

    float *soil_heat_d;
    float *net_radiation_d;
    float *surface_temperature_d;

    float *enb_d;
    float *eo_d;
    float *ea_d;
    float *short_wave_radiation_d;
    float *large_wave_radiation_surface_d;
    float *large_wave_radiation_atmosphere_d;

    float *d0_d;
    float *kb1_d;
    float *zom_d;
    float *ustar_d;
    float *rah_d;

    float *sensible_heat_flux_d;
    float *latent_heat_flux_d;
    float *net_radiation_24h_d;
    float *evapotranspiration_fraction_d;
    float *sensible_heat_flux_24h_d;
    float *latent_heat_flux_24h_d;
    float *evapotranspiration_24h_d;
    float *evapotranspiration_d;

    /**
     * @brief  Constructor.
     */
    Products();

    /**
     * @brief  Constructor.
     * @param  width_band: Band width.
     * @param  height_band: Band height.
     */
    Products(uint32_t width_band, uint32_t height_band);

    /**
     * @brief Read the data and move to the device.
     *
     * @param  landsat_bands: Array with the TIFF files.
     * @return string with the time spent.
     */
    string read_data(TIFF **landsat_bands);

    /**
     * @brief Copy the products to the host.
     *
     * @return string with the time spent.
     */
    string host_data();

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
     * @param  products: Products struct.
     * @return string with the time spent.
     */
    string select_endmembers(Products products);

    /**
     * @brief make the rah cycle converge
     *
     * @param  station: Station struct.
     * @param  model_method: Method to converge the rah cycle.
     *
     * @return string with the time spent.
     */
    string converge_rah_cycle(Products products, Station station);

    /**
     * @brief Compute the final products.
     *
     * @param  station: Station struct.
     * @return string with the time spent.
     */
    string compute_H_ET(Products products, Station station, MTL mtl);

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