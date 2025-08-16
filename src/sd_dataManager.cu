#include "cuda_utils.h"
#include "kernels.cuh"
#include "surfaceData.cuh"

Products::Products(uint32_t width_band, uint32_t height_band)
{
    this->width_band = width_band;
    this->height_band = height_band;
    this->band_bytes = height_band * width_band * sizeof(float);

    this->band_blue = (float *)malloc(band_bytes);
    this->band_green = (float *)malloc(band_bytes);
    this->band_red = (float *)malloc(band_bytes);
    this->band_nir = (float *)malloc(band_bytes);
    this->band_swir1 = (float *)malloc(band_bytes);
    this->band_termal = (float *)malloc(band_bytes);
    this->band_swir2 = (float *)malloc(band_bytes);
    this->tal = (float *)malloc(band_bytes);

    this->radiance_blue = (float *)malloc(band_bytes);
    this->radiance_green = (float *)malloc(band_bytes);
    this->radiance_red = (float *)malloc(band_bytes);
    this->radiance_nir = (float *)malloc(band_bytes);
    this->radiance_swir1 = (float *)malloc(band_bytes);
    this->radiance_termal = (float *)malloc(band_bytes);
    this->radiance_swir2 = (float *)malloc(band_bytes);

    this->reflectance_blue = (float *)malloc(band_bytes);
    this->reflectance_green = (float *)malloc(band_bytes);
    this->reflectance_red = (float *)malloc(band_bytes);
    this->reflectance_nir = (float *)malloc(band_bytes);
    this->reflectance_swir1 = (float *)malloc(band_bytes);
    this->reflectance_termal = (float *)malloc(band_bytes);
    this->reflectance_swir2 = (float *)malloc(band_bytes);

    this->albedo = (float *)malloc(band_bytes);
    this->ndvi = (float *)malloc(band_bytes);
    this->soil_heat = (float *)malloc(band_bytes);
    this->surface_temperature = (float *)malloc(band_bytes);
    this->net_radiation = (float *)malloc(band_bytes);
    this->lai = (float *)malloc(band_bytes);
    this->savi = (float *)malloc(band_bytes);
    this->pai = (float *)malloc(band_bytes);
    this->enb_emissivity = (float *)malloc(band_bytes);
    this->eo_emissivity = (float *)malloc(band_bytes);
    this->ea_emissivity = (float *)malloc(band_bytes);
    this->short_wave_radiation = (float *)malloc(band_bytes);
    this->large_wave_radiation_surface = (float *)malloc(band_bytes);
    this->large_wave_radiation_atmosphere = (float *)malloc(band_bytes);

    this->surface_temperature = (float *)malloc(band_bytes);
    this->d0 = (float *)malloc(band_bytes);
    this->zom = (float *)malloc(band_bytes);
    this->ustar = (float *)malloc(band_bytes);
    this->kb1 = (float *)malloc(band_bytes);
    this->aerodynamic_resistance = (float *)malloc(band_bytes);
    this->sensible_heat_flux = (float *)malloc(band_bytes);

    this->latent_heat_flux = (float *)malloc(band_bytes);
    this->net_radiation_24h = (float *)malloc(band_bytes);
    this->evapotranspiration_24h = (float *)malloc(band_bytes);

    HANDLE_ERROR(cudaMemcpyToSymbol(width_d, &width_band, sizeof(int), 0, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpyToSymbol(height_d, &height_band, sizeof(int), 0, cudaMemcpyHostToDevice));

    const size_t MAXC = sizeof(Endmember) * height_band * width_band;
    HANDLE_ERROR(cudaMalloc((void **)&this->hotCandidates_d, MAXC));
    HANDLE_ERROR(cudaMalloc((void **)&this->coldCandidates_d, MAXC));

    HANDLE_ERROR(cudaMalloc((void **)&this->band_blue_d, band_bytes));
    HANDLE_ERROR(cudaMalloc((void **)&this->band_green_d, band_bytes));
    HANDLE_ERROR(cudaMalloc((void **)&this->band_red_d, band_bytes));
    HANDLE_ERROR(cudaMalloc((void **)&this->band_nir_d, band_bytes));
    HANDLE_ERROR(cudaMalloc((void **)&this->band_swir1_d, band_bytes));
    HANDLE_ERROR(cudaMalloc((void **)&this->band_termal_d, band_bytes));
    HANDLE_ERROR(cudaMalloc((void **)&this->band_swir2_d, band_bytes));
    HANDLE_ERROR(cudaMalloc((void **)&this->tal_d, band_bytes));

    HANDLE_ERROR(cudaMalloc((void **)&this->radiance_blue_d, band_bytes));
    HANDLE_ERROR(cudaMalloc((void **)&this->radiance_green_d, band_bytes));
    HANDLE_ERROR(cudaMalloc((void **)&this->radiance_red_d, band_bytes));
    HANDLE_ERROR(cudaMalloc((void **)&this->radiance_nir_d, band_bytes));
    HANDLE_ERROR(cudaMalloc((void **)&this->radiance_swir1_d, band_bytes));
    HANDLE_ERROR(cudaMalloc((void **)&this->radiance_termal_d, band_bytes));
    HANDLE_ERROR(cudaMalloc((void **)&this->radiance_swir2_d, band_bytes));

    HANDLE_ERROR(cudaMalloc((void **)&this->reflectance_blue_d, band_bytes));
    HANDLE_ERROR(cudaMalloc((void **)&this->reflectance_green_d, band_bytes));
    HANDLE_ERROR(cudaMalloc((void **)&this->reflectance_red_d, band_bytes));
    HANDLE_ERROR(cudaMalloc((void **)&this->reflectance_nir_d, band_bytes));
    HANDLE_ERROR(cudaMalloc((void **)&this->reflectance_swir1_d, band_bytes));
    HANDLE_ERROR(cudaMalloc((void **)&this->reflectance_termal_d, band_bytes));
    HANDLE_ERROR(cudaMalloc((void **)&this->reflectance_swir2_d, band_bytes));

    HANDLE_ERROR(cudaMalloc((void **)&this->albedo_d, band_bytes));
    HANDLE_ERROR(cudaMalloc((void **)&this->ndvi_d, band_bytes));
    HANDLE_ERROR(cudaMalloc((void **)&this->pai_d, band_bytes));
    HANDLE_ERROR(cudaMalloc((void **)&this->lai_d, band_bytes));

    HANDLE_ERROR(cudaMalloc((void **)&this->enb_d, band_bytes));
    HANDLE_ERROR(cudaMalloc((void **)&this->eo_d, band_bytes));
    HANDLE_ERROR(cudaMalloc((void **)&this->ea_d, band_bytes));
    HANDLE_ERROR(cudaMalloc((void **)&this->short_wave_radiation_d, band_bytes));
    HANDLE_ERROR(cudaMalloc((void **)&this->large_wave_radiation_surface_d, band_bytes));
    HANDLE_ERROR(cudaMalloc((void **)&this->large_wave_radiation_atmosphere_d, band_bytes));
    HANDLE_ERROR(cudaMalloc((void **)&this->surface_temperature_d, band_bytes));
    HANDLE_ERROR(cudaMalloc((void **)&this->net_radiation_d, band_bytes));
    HANDLE_ERROR(cudaMalloc((void **)&this->soil_heat_d, band_bytes));

    HANDLE_ERROR(cudaMalloc((void **)&this->zom_d, band_bytes));
    HANDLE_ERROR(cudaMalloc((void **)&this->d0_d, band_bytes));
    HANDLE_ERROR(cudaMalloc((void **)&this->kb1_d, band_bytes));
    HANDLE_ERROR(cudaMalloc((void **)&this->ustar_d, band_bytes));
    HANDLE_ERROR(cudaMalloc((void **)&this->rah_d, band_bytes));
    HANDLE_ERROR(cudaMalloc((void **)&this->sensible_heat_flux_d, band_bytes));

    HANDLE_ERROR(cudaMalloc((void **)&this->latent_heat_flux_d, band_bytes));
    HANDLE_ERROR(cudaMalloc((void **)&this->net_radiation_24h_d, band_bytes));
    HANDLE_ERROR(cudaMalloc((void **)&this->evapotranspiration_24h_d, band_bytes));
};

string Products::read_data(TIFF **landsat_bands)
{
    system_clock::time_point begin, end;
    int64_t initial_time, final_time;
    float general_time;

    begin = system_clock::now();
    initial_time = duration_cast<nanoseconds>(begin.time_since_epoch()).count();

    const int num_bands = INPUT_BAND_ELEV_INDEX;

    for (int i = 0; i < num_bands; i++) {
        TIFF *curr_band = landsat_bands[i];
        tstrip_t num_strips = TIFFNumberOfStrips(curr_band);
        size_t offset = 0;

        tsize_t strip_size = TIFFStripSize(curr_band);
        tdata_t strip_buffer = _TIFFmalloc(strip_size);
        if (!strip_buffer) throw std::runtime_error("Erro alocando strip_buffer");

        float* band_ptr = nullptr;
        switch (i) {
            case 0: band_ptr = this->band_blue;  break;
            case 1: band_ptr = this->band_green; break;
            case 2: band_ptr = this->band_red;   break;
            case 3: band_ptr = this->band_nir;   break;
            case 4: band_ptr = this->band_swir1; break;
            case 5: band_ptr = this->band_termal;break;
            case 6: band_ptr = this->band_swir2; break;
            case 7: band_ptr = this->tal;        break;
        }

        for (tstrip_t strip = 0; strip < num_strips; strip++) {
            tsize_t bytes_read = TIFFReadEncodedStrip(curr_band, strip, strip_buffer, strip_size);
            if (bytes_read == -1) {
                _TIFFfree(strip_buffer);
                throw std::runtime_error("Erro lendo strip");
            }

            unsigned int pixels_in_strip = bytes_read / sizeof(float);
            float* strip_data = static_cast<float*>(strip_buffer);

            if (i != 7) {
                memcpy(band_ptr + offset, strip_data, bytes_read);
            } else {
                for (unsigned int p = 0; p < pixels_in_strip; p++) {
                    band_ptr[offset + p] = 0.75f + 2.0f * powf(10.0f, -5.0f) * strip_data[p];
                }
            }

            offset += pixels_in_strip;
        }

        _TIFFfree(strip_buffer);
    }

    // Copia os dados para a GPU
    HANDLE_ERROR(cudaMemcpy(band_blue_d,  band_blue,  band_bytes, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(band_green_d, band_green, band_bytes, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(band_red_d,   band_red,   band_bytes, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(band_nir_d,   band_nir,   band_bytes, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(band_swir1_d, band_swir1, band_bytes, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(band_termal_d,band_termal,band_bytes, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(band_swir2_d, band_swir2, band_bytes, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(tal_d,        tal,        band_bytes, cudaMemcpyHostToDevice));

    end = system_clock::now();
    final_time = duration_cast<nanoseconds>(end.time_since_epoch()).count();
    general_time = duration_cast<nanoseconds>(end - begin).count() / 1000.0f / 1000.0f;

    return "SERIAL,P0_READ_INPUT," + to_string(general_time) + "," + to_string(initial_time) + "," + to_string(final_time) + "\n";
}

string Products::host_data()
{
    system_clock::time_point begin, end;
    float general_time;
    int64_t initial_time, final_time;

    begin = system_clock::now();
    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    // HANDLE_ERROR(cudaMemcpy(radiance_blue, radiance_blue_d, band_bytes, cudaMemcpyDeviceToHost));
    // HANDLE_ERROR(cudaMemcpy(radiance_green, radiance_green_d, band_bytes, cudaMemcpyDeviceToHost));
    // HANDLE_ERROR(cudaMemcpy(radiance_red, radiance_red_d, band_bytes, cudaMemcpyDeviceToHost));
    // HANDLE_ERROR(cudaMemcpy(radiance_nir, radiance_nir_d, band_bytes, cudaMemcpyDeviceToHost));
    // HANDLE_ERROR(cudaMemcpy(radiance_swir1, radiance_swir1_d, band_bytes, cudaMemcpyDeviceToHost));
    // HANDLE_ERROR(cudaMemcpy(radiance_termal, radiance_termal_d, band_bytes, cudaMemcpyDeviceToHost));
    // HANDLE_ERROR(cudaMemcpy(radiance_swir2, radiance_swir2_d, band_bytes, cudaMemcpyDeviceToHost));
    // HANDLE_ERROR(cudaMemcpy(reflectance_blue, reflectance_blue_d, band_bytes, cudaMemcpyDeviceToHost));
    // HANDLE_ERROR(cudaMemcpy(reflectance_green, reflectance_green_d, band_bytes, cudaMemcpyDeviceToHost));
    // HANDLE_ERROR(cudaMemcpy(reflectance_red, reflectance_red_d, band_bytes, cudaMemcpyDeviceToHost));
    // HANDLE_ERROR(cudaMemcpy(reflectance_nir, reflectance_nir_d, band_bytes, cudaMemcpyDeviceToHost));
    // HANDLE_ERROR(cudaMemcpy(reflectance_swir1, reflectance_swir1_d, band_bytes, cudaMemcpyDeviceToHost));
    // HANDLE_ERROR(cudaMemcpy(reflectance_termal, reflectance_termal_d, band_bytes, cudaMemcpyDeviceToHost));
    // HANDLE_ERROR(cudaMemcpy(reflectance_swir2, reflectance_swir2_d, band_bytes, cudaMemcpyDeviceToHost));
    // HANDLE_ERROR(cudaMemcpy(albedo, albedo_d, band_bytes, cudaMemcpyDeviceToHost));
    // HANDLE_ERROR(cudaMemcpy(ndvi, ndvi_d, band_bytes, cudaMemcpyDeviceToHost));
    // HANDLE_ERROR(cudaMemcpy(pai, pai_d, band_bytes, cudaMemcpyDeviceToHost));
    // HANDLE_ERROR(cudaMemcpy(lai, lai_d, band_bytes, cudaMemcpyDeviceToHost));
    // HANDLE_ERROR(cudaMemcpy(enb_emissivity, enb_d, band_bytes, cudaMemcpyDeviceToHost));
    // HANDLE_ERROR(cudaMemcpy(eo_emissivity, eo_d, band_bytes, cudaMemcpyDeviceToHost));
    // HANDLE_ERROR(cudaMemcpy(ea_emissivity, ea_d, band_bytes, cudaMemcpyDeviceToHost));
    // HANDLE_ERROR(cudaMemcpy(surface_temperature, surface_temperature_d, band_bytes, cudaMemcpyDeviceToHost));
    // HANDLE_ERROR(cudaMemcpy(short_wave_radiation, short_wave_radiation_d, band_bytes, cudaMemcpyDeviceToHost));
    // HANDLE_ERROR(cudaMemcpy(large_wave_radiation_surface, large_wave_radiation_surface_d, band_bytes, cudaMemcpyDeviceToHost));
    // HANDLE_ERROR(cudaMemcpy(large_wave_radiation_atmosphere, large_wave_radiation_atmosphere_d, band_bytes, cudaMemcpyDeviceToHost));
    // HANDLE_ERROR(cudaMemcpy(net_radiation, net_radiation_d, band_bytes, cudaMemcpyDeviceToHost));
    // HANDLE_ERROR(cudaMemcpy(soil_heat, soil_heat_d, band_bytes, cudaMemcpyDeviceToHost));
    // HANDLE_ERROR(cudaMemcpy(d0, d0_d, band_bytes, cudaMemcpyDeviceToHost));
    // HANDLE_ERROR(cudaMemcpy(zom, zom_d, band_bytes, cudaMemcpyDeviceToHost));
    // HANDLE_ERROR(cudaMemcpy(ustar, ustar_d, band_bytes, cudaMemcpyDeviceToHost));
    // HANDLE_ERROR(cudaMemcpy(kb1, kb1_d, band_bytes, cudaMemcpyDeviceToHost));
    // HANDLE_ERROR(cudaMemcpy(aerodynamic_resistance, rah_d, band_bytes, cudaMemcpyDeviceToHost));
    // HANDLE_ERROR(cudaMemcpy(sensible_heat_flux, sensible_heat_flux_d, band_bytes, cudaMemcpyDeviceToHost));
    // HANDLE_ERROR(cudaMemcpy(latent_heat_flux, latent_heat_flux_d, band_bytes, cudaMemcpyDeviceToHost));
    // HANDLE_ERROR(cudaMemcpy(net_radiation_24h, net_radiation_24h_d, band_bytes, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(evapotranspiration_24h, evapotranspiration_24h_d, band_bytes, cudaMemcpyDeviceToHost));

    end = system_clock::now();
    general_time = duration_cast<nanoseconds>(end - begin).count() / 1000000.0;
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    return "SERIAL,P5_COPY_HOST," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
}

string Products::save_products(string output_path)
{
    system_clock::time_point begin, end;
    float general_time;
    int64_t initial_time, final_time;

    begin = system_clock::now();
    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    // saveTiff(output_path + "/albedo.tif", albedo, height_band, width_band);
    // saveTiff(output_path + "/ndvi.tif", ndvi, height_band, width_band);
    // saveTiff(output_path + "/pai.tif", pai, height_band, width_band);
    // saveTiff(output_path + "/lai.tif", lai, height_band, width_band);
    // saveTiff(output_path + "/enb_emissivity.tif", enb_emissivity, height_band, width_band);
    // saveTiff(output_path + "/eo_emissivity.tif", eo_emissivity, height_band, width_band);
    // saveTiff(output_path + "/ea_emissivity.tif", ea_emissivity, height_band, width_band);
    // saveTiff(output_path + "/surface_temperature.tif", surface_temperature, height_band, width_band);
    // saveTiff(output_path + "/net_radiation.tif", net_radiation, height_band, width_band);
    // saveTiff(output_path + "/soil_heat_flux.tif", soil_heat, height_band, width_band);
    // saveTiff(output_path + "/d0.tif", d0, height_band, width_band);
    // saveTiff(output_path + "/zom.tif", zom, height_band, width_band);
    // saveTiff(output_path + "/ustar.tif", ustar, height_band, width_band);
    // saveTiff(output_path + "/kb.tif", kb1, height_band, width_band);
    // saveTiff(output_path + "/rah.tif", aerodynamic_resistance, height_band, width_band);
    // saveTiff(output_path + "/sensible_heat_flux.tif", sensible_heat_flux, height_band, width_band);
    // saveTiff(output_path + "/latent_heat_flux.tif", latent_heat_flux, height_band, width_band);
    // saveTiff(output_path + "/net_radiation_24h.tif", net_radiation_24h, height_band, width_band);
    saveTiff(output_path + "/evapotranspiration_24h.tif", evapotranspiration_24h, height_band, width_band);

    end = system_clock::now();
    general_time = duration_cast<nanoseconds>(end - begin).count() / 1000000.0;
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    return "SERIAL,P6_SAVE_PRODS," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::print_products(string output_path)
{
    system_clock::time_point begin, end;
    float general_time;
    int64_t initial_time, final_time;

    begin = system_clock::now();
    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    // redirect the stdout to a file]
    std::ofstream out(output_path + "/products.txt");
    std::streambuf *coutbuf = std::cout.rdbuf();
    std::cout.rdbuf(out.rdbuf());

//    std::cout << "==== Albedo" << std::endl;
//    printLinearPointer(albedo, height_band, width_band);
//
//    std::cout << "==== NDVI" << std::endl;
//    printLinearPointer(ndvi, height_band, width_band);
//
//    std::cout << "==== PAI" << std::endl;
//    printLinearPointer(pai, height_band, width_band);
//
//    std::cout << "==== LAI" << std::endl;
//    printLinearPointer(lai, height_band, width_band);
//
//    std::cout << "==== ENB Emissivity" << std::endl;
//    printLinearPointer(enb_emissivity, height_band, width_band);
//
//    std::cout << "==== EO Emissivity" << std::endl;
//    printLinearPointer(eo_emissivity, height_band, width_band);
//
//    std::cout << "==== EA Emissivity" << std::endl;
//    printLinearPointer(ea_emissivity, height_band, width_band);
//
//    std::cout << "==== Surface Temperature" << std::endl;
//    printLinearPointer(surface_temperature, height_band, width_band);
//
//    std::cout << "==== Net Radiation" << std::endl;
//    printLinearPointer(net_radiation, height_band, width_band);
//
//    std::cout << "==== Soil Heat Flux" << std::endl;
//    printLinearPointer(soil_heat, height_band, width_band);
//
//    std::cout << "==== D0" << std::endl;
//    printLinearPointer(d0, height_band, width_band);
//
//    std::cout << "==== ZOM" << std::endl;
//    printLinearPointer(zom, height_band, width_band);
//
//    std::cout << "==== Ustar" << std::endl;
//    printLinearPointer(ustar, height_band, width_band);
//
//    std::cout << "==== KB" << std::endl;
//    printLinearPointer(kb1, height_band, width_band);
//
//    std::cout << "==== RAH" << std::endl;
//    printLinearPointer(aerodynamic_resistance, height_band, width_band);
//
//    std::cout << "==== Sensible Heat Flux" << std::endl;
//    printLinearPointer(sensible_heat_flux, height_band, width_band);
//
//    std::cout << "==== Latent Heat Flux" << std::endl;
//    printLinearPointer(latent_heat_flux, height_band, width_band);
//
//    std::cout << "==== Net Radiation 24h" << std::endl;
//    printLinearPointer(net_radiation_24h, height_band, width_band);
//
    std::cout << "==== Evapotranspiration 24h" << std::endl;
    printLinearPointer(evapotranspiration_24h, height_band, width_band);

    end = system_clock::now();
    general_time = duration_cast<nanoseconds>(end - begin).count() / 1000000.0;
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    return "SERIAL,P7_STDOUT_PRODS," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

void Products::close(TIFF **landsat_bands)
{
    for (int i = 1; i < 8; i++) {
        TIFFClose(landsat_bands[i]);
    }

    free(this->band_blue);
    free(this->band_green);
    free(this->band_red);
    free(this->band_nir);
    free(this->band_swir1);
    free(this->band_termal);
    free(this->band_swir2);
    free(this->tal);

    free(this->radiance_blue);
    free(this->radiance_green);
    free(this->radiance_red);
    free(this->radiance_nir);
    free(this->radiance_swir1);
    free(this->radiance_termal);
    free(this->radiance_swir2);

    free(this->reflectance_blue);
    free(this->reflectance_green);
    free(this->reflectance_red);
    free(this->reflectance_nir);
    free(this->reflectance_swir1);
    free(this->reflectance_termal);
    free(this->reflectance_swir2);

    free(this->albedo);
    free(this->ndvi);
    free(this->lai);
    free(this->pai);

    free(this->enb_emissivity);
    free(this->eo_emissivity);
    free(this->ea_emissivity);
    free(this->short_wave_radiation);
    free(this->large_wave_radiation_surface);
    free(this->large_wave_radiation_atmosphere);

    free(this->surface_temperature);
    free(this->net_radiation);
    free(this->soil_heat);

    free(this->zom);
    free(this->d0);
    free(this->ustar);
    free(this->kb1);
    free(this->aerodynamic_resistance);
    free(this->sensible_heat_flux);

    free(this->latent_heat_flux);
    free(this->net_radiation_24h);
    free(this->evapotranspiration_24h);

    HANDLE_ERROR(cudaFree(this->band_blue_d));
    HANDLE_ERROR(cudaFree(this->band_green_d));
    HANDLE_ERROR(cudaFree(this->band_red_d));
    HANDLE_ERROR(cudaFree(this->band_nir_d));
    HANDLE_ERROR(cudaFree(this->band_swir1_d));
    HANDLE_ERROR(cudaFree(this->band_termal_d));
    HANDLE_ERROR(cudaFree(this->band_swir2_d));
    HANDLE_ERROR(cudaFree(this->tal_d));

    HANDLE_ERROR(cudaFree(this->radiance_blue_d));
    HANDLE_ERROR(cudaFree(this->radiance_green_d));
    HANDLE_ERROR(cudaFree(this->radiance_red_d));
    HANDLE_ERROR(cudaFree(this->radiance_nir_d));
    HANDLE_ERROR(cudaFree(this->radiance_swir1_d));
    HANDLE_ERROR(cudaFree(this->radiance_termal_d));
    HANDLE_ERROR(cudaFree(this->radiance_swir2_d));

    HANDLE_ERROR(cudaFree(this->reflectance_blue_d));
    HANDLE_ERROR(cudaFree(this->reflectance_green_d));
    HANDLE_ERROR(cudaFree(this->reflectance_red_d));
    HANDLE_ERROR(cudaFree(this->reflectance_nir_d));
    HANDLE_ERROR(cudaFree(this->reflectance_swir1_d));
    HANDLE_ERROR(cudaFree(this->reflectance_termal_d));
    HANDLE_ERROR(cudaFree(this->reflectance_swir2_d));

    HANDLE_ERROR(cudaFree(this->albedo_d));
    HANDLE_ERROR(cudaFree(this->ndvi_d));
    HANDLE_ERROR(cudaFree(this->pai_d));
    HANDLE_ERROR(cudaFree(this->lai_d));

    HANDLE_ERROR(cudaFree(this->enb_d));
    HANDLE_ERROR(cudaFree(this->eo_d));
    HANDLE_ERROR(cudaFree(this->ea_d));
    HANDLE_ERROR(cudaFree(this->short_wave_radiation_d));
    HANDLE_ERROR(cudaFree(this->large_wave_radiation_surface_d));
    HANDLE_ERROR(cudaFree(this->large_wave_radiation_atmosphere_d));

    HANDLE_ERROR(cudaFree(this->surface_temperature_d));
    HANDLE_ERROR(cudaFree(this->net_radiation_d));
    HANDLE_ERROR(cudaFree(this->soil_heat_d));

    HANDLE_ERROR(cudaFree(this->zom_d));
    HANDLE_ERROR(cudaFree(this->d0_d));
    HANDLE_ERROR(cudaFree(this->kb1_d));
    HANDLE_ERROR(cudaFree(this->ustar_d));
    HANDLE_ERROR(cudaFree(this->rah_d));
    HANDLE_ERROR(cudaFree(this->sensible_heat_flux_d));

    HANDLE_ERROR(cudaFree(this->latent_heat_flux_d));
    HANDLE_ERROR(cudaFree(this->net_radiation_24h_d));
    HANDLE_ERROR(cudaFree(this->evapotranspiration_24h_d));
};