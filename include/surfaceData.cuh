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
    half *band_blue;
    half *band_green;
    half *band_red;
    half *band_nir;
    half *band_swir1;
    half *band_termal;
    half *band_swir2;
    half *tal;

    half *radiance_blue;
    half *radiance_green;
    half *radiance_red;
    half *radiance_nir;
    half *radiance_swir1;
    half *radiance_termal;
    half *radiance_swir2;

    half *reflectance_blue;
    half *reflectance_green;
    half *reflectance_red;
    half *reflectance_nir;
    half *reflectance_swir1;
    half *reflectance_termal;
    half *reflectance_swir2;

    half *albedo;
    half *ndvi;
    half *savi;
    half *lai;
    half *pai;

    half *soil_heat;
    half *net_radiation;
    half *surface_temperature;

    half *enb_emissivity;
    half *eo_emissivity;
    half *ea_emissivity;
    half *short_wave_radiation;
    half *large_wave_radiation_surface;
    half *large_wave_radiation_atmosphere;

    half *d0;
    half *zom;
    half *ustar;
    half *kb1;
    half *aerodynamic_resistance;
    half *sensible_heat_flux;

    half *latent_heat_flux;
    half *net_radiation_24h;
    half *evapotranspiration_fraction;
    half *sensible_heat_flux_24h;
    half *latent_heat_flux_24h;
    half *evapotranspiration_24h;
    half *evapotranspiration;

    // Device pointers
    half *band_blue_d;
    half *band_green_d;
    half *band_red_d;
    half *band_nir_d;
    half *band_swir1_d;
    half *band_termal_d;
    half *band_swir2_d;
    half *tal_d;

    half *radiance_blue_d;
    half *radiance_green_d;
    half *radiance_red_d;
    half *radiance_nir_d;
    half *radiance_swir1_d;
    half *radiance_termal_d;
    half *radiance_swir2_d;

    half *reflectance_blue_d;
    half *reflectance_green_d;
    half *reflectance_red_d;
    half *reflectance_nir_d;
    half *reflectance_swir1_d;
    half *reflectance_termal_d;
    half *reflectance_swir2_d;

    half *albedo_d;
    half *ndvi_d;
    half *pai_d;
    half *savi_d;
    half *lai_d;

    half *soil_heat_d;
    half *net_radiation_d;
    half *surface_temperature_d;

    half *enb_d;
    half *eo_d;
    half *ea_d;
    half *short_wave_radiation_d;
    half *large_wave_radiation_surface_d;
    half *large_wave_radiation_atmosphere_d;

    half *d0_d;
    half *kb1_d;
    half *zom_d;
    half *ustar_d;
    half *rah_d;

    half *sensible_heat_flux_d;
    half *latent_heat_flux_d;
    half *net_radiation_24h_d;
    half *evapotranspiration_fraction_d;
    half *sensible_heat_flux_24h_d;
    half *latent_heat_flux_24h_d;
    half *evapotranspiration_24h_d;
    half *evapotranspiration_d;

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
     * @param  products: Products struct.
     * @param  station: Station struct.
     * @param  mtl: MTL struct.
     * 
     * @return string with the time spent.
     */
    string compute_Rn_G(Products products, Station station, MTL mtl);

    /**
     * @brief Select the cold and hot endmembers
     *
     * @param  products: Products struct.
     * 
     * @return string with the time spent.
     */
    string select_endmembers(Products products);

    /**
     * @brief make the rah cycle converge
     *
     * @param  products: Products struct.
     * @param  station: Station struct.
     *
     * @return string with the time spent.
     */
    string converge_rah_cycle(Products products, Station station);

    /**
     * @brief Compute the final products.
     *
     * @param  products: Products struct.
     * @param  station: Station struct.
     * @param  mtl: MTL struct.
     * 
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