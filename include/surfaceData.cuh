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
    int band_bytes_double;

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

    double *radiance_blue;
    double *radiance_green;
    double *radiance_red;
    double *radiance_nir;
    double *radiance_swir1;
    double *radiance_termal;
    double *radiance_swir2;

    double *reflectance_blue;
    double *reflectance_green;
    double *reflectance_red;
    double *reflectance_nir;
    double *reflectance_swir1;
    double *reflectance_termal;
    double *reflectance_swir2;

    double *albedo;
    double *ndvi;
    double *savi;
    double *lai;
    double *pai;

    double *soil_heat;
    double *net_radiation;
    double *surface_temperature;

    double *enb_emissivity;
    double *eo_emissivity;
    double *ea_emissivity;
    double *short_wave_radiation;
    double *large_wave_radiation_surface;
    double *large_wave_radiation_atmosphere;

    double *d0;
    double *zom;
    double *ustar;
    double *kb1;
    double *aerodynamic_resistance;
    double *sensible_heat_flux;

    double *latent_heat_flux;
    double *net_radiation_24h;
    double *evapotranspiration_fraction;
    double *sensible_heat_flux_24h;
    double *latent_heat_flux_24h;
    double *evapotranspiration_24h;
    double *evapotranspiration;

    // Device pointers
    float *band_blue_d;
    float *band_green_d;
    float *band_red_d;
    float *band_nir_d;
    float *band_swir1_d;
    float *band_termal_d;
    float *band_swir2_d;
    float *tal_d;

    double *radiance_blue_d;
    double *radiance_green_d;
    double *radiance_red_d;
    double *radiance_nir_d;
    double *radiance_swir1_d;
    double *radiance_termal_d;
    double *radiance_swir2_d;

    double *reflectance_blue_d;
    double *reflectance_green_d;
    double *reflectance_red_d;
    double *reflectance_nir_d;
    double *reflectance_swir1_d;
    double *reflectance_termal_d;
    double *reflectance_swir2_d;

    double *albedo_d;
    double *ndvi_d;
    double *pai_d;
    double *savi_d;
    double *lai_d;

    double *soil_heat_d;
    double *net_radiation_d;
    double *surface_temperature_d;

    double *enb_d;
    double *eo_d;
    double *ea_d;
    double *short_wave_radiation_d;
    double *large_wave_radiation_surface_d;
    double *large_wave_radiation_atmosphere_d;

    double *d0_d;
    double *kb1_d;
    double *zom_d;
    double *ustar_d;
    double *rah_d;

    double *sensible_heat_flux_d;
    double *latent_heat_flux_d;
    double *net_radiation_24h_d;
    double *evapotranspiration_fraction_d;
    double *sensible_heat_flux_24h_d;
    double *latent_heat_flux_24h_d;
    double *evapotranspiration_24h_d;
    double *evapotranspiration_d;
    
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
