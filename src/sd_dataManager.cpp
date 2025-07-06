#include "surfaceData.h"
#include "thread_utils.h"

Products::Products(uint32_t width_band, uint32_t height_band)
{
    this->width_band = width_band;
    this->height_band = height_band;
    this->band_bytes = height_band * width_band * sizeof(float);

    this->hotEndmemberPos = (int *)malloc(sizeof(int) * 2);
    this->coldEndmemberPos = (int *)malloc(sizeof(int) * 2);

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
};


string Products::read_data(TIFF **landsat_bands)
{
    system_clock::time_point begin, end;
    int64_t initial_time, final_time;
    float general_time;

    begin = system_clock::now();
    initial_time = duration_cast<nanoseconds>(begin.time_since_epoch()).count();

    const int num_bands = INPUT_BAND_ELEV_INDEX; // 8

    parallel_for(num_bands, [&](int start_band_idx, int end_band_idx) {
        for (int i = start_band_idx; i < end_band_idx; i++) {
            TIFF *curr_band = landsat_bands[i];

            // Número de strips e tamanho total dos dados na banda
        tstrip_t strips_per_band = TIFFNumberOfStrips(curr_band);

        size_t strip_size = 0;
        tdata_t strip_buffer = nullptr;

        size_t offset = 0; // offset em bytes no buffer final da banda

        for (tstrip_t strip = 0; strip < strips_per_band; strip++) {
            strip_size = TIFFStripSize(curr_band);

            if (!strip_buffer) {
                strip_buffer = (tdata_t) _TIFFmalloc(strip_size);
                if (!strip_buffer) {
                    // tratamento de erro alocação
                    throw std::runtime_error("Erro alocando buffer de strip");
                }
            }

            // Leitura do strip (bloco grande)
            if (TIFFReadEncodedStrip(curr_band, strip, strip_buffer, strip_size) == -1) {
                _TIFFfree(strip_buffer);
                throw std::runtime_error("Erro lendo strip");
            }

            // Número de pixels no strip
            // Obs: TIFFScanlineSize dá tamanho em bytes de uma linha, precisa converter para pixels
            unsigned int bytes_per_pixel = sizeof(float); // assumindo float armazenado (confirme no TIFF!)
            unsigned int pixels_per_strip = strip_size / bytes_per_pixel;

            // Agora copia e converte valores para o buffer final (float*)
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

            if (i != 7) {
                // para bandas normais: copiar diretamente
                memcpy(band_ptr + offset / bytes_per_pixel, strip_buffer, strip_size);
            } else {
                // para banda tal: aplicar fórmula especial por pixel
                // precisamos percorrer os floats do strip_buffer
                float* strip_float = (float*) strip_buffer;
                for (unsigned int p = 0; p < pixels_per_strip; p++) {
                    band_ptr[offset/bytes_per_pixel + p] = 0.75f + 2.0f * powf(10.0f, -5.0f) * strip_float[p];
                }
            }

            offset += strip_size;
        }

        if (strip_buffer) {
            _TIFFfree(strip_buffer);
            strip_buffer = nullptr;
        }
    } // End of inner loop (bands for this thread)
}); // End of parallel_for

    end = system_clock::now();
    general_time = duration_cast<nanoseconds>(end - begin).count() / 1e6;
    final_time = duration_cast<nanoseconds>(end.time_since_epoch()).count();

    return "PARALLEL,P0_READ_INPUT," + to_string(general_time) + "," + to_string(initial_time) + "," + to_string(final_time) + "\n";
}

string Products::save_products(string output_path)
{
    system_clock::time_point begin, end;
    float general_time;
    int64_t initial_time, final_time;

    begin = system_clock::now();    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

//    saveTiff(output_path + "/albedo.tif", albedo, height_band, width_band);
//    saveTiff(output_path + "/ndvi.tif", ndvi, height_band, width_band);
//    saveTiff(output_path + "/pai.tif", pai, height_band, width_band);
//    saveTiff(output_path + "/lai.tif", lai, height_band, width_band);
//    saveTiff(output_path + "/enb_emissivity.tif", enb_emissivity, height_band, width_band);
//    saveTiff(output_path + "/eo_emissivity.tif", eo_emissivity, height_band, width_band);
//    saveTiff(output_path + "/ea_emissivity.tif", ea_emissivity, height_band, width_band);
//    saveTiff(output_path + "/surface_temperature.tif", surface_temperature, height_band, width_band);
//    saveTiff(output_path + "/net_radiation.tif", net_radiation, height_band, width_band);
//    saveTiff(output_path + "/soil_heat_flux.tif", soil_heat, height_band, width_band);
//    saveTiff(output_path + "/d0.tif", d0, height_band, width_band);
//    saveTiff(output_path + "/zom.tif", zom, height_band, width_band);
//    saveTiff(output_path + "/ustar.tif", ustar, height_band, width_band);
//    saveTiff(output_path + "/kb.tif", kb1, height_band, width_band);
//    saveTiff(output_path + "/rah.tif", aerodynamic_resistance, height_band, width_band);
//    saveTiff(output_path + "/sensible_heat_flux.tif", sensible_heat_flux, height_band, width_band);
//    saveTiff(output_path + "/latent_heat_flux.tif", latent_heat_flux, height_band, width_band);
//    saveTiff(output_path + "/net_radiation_24h.tif", net_radiation_24h, height_band, width_band);
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

    begin = system_clock::now();    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    // redirect the stdout to a file]
    std::ofstream out(output_path + "/products.txt");
    std::streambuf *coutbuf = std::cout.rdbuf();
    std::cout.rdbuf(out.rdbuf());
//
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
};
