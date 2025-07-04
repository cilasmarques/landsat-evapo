#include "cuda_utils.h"
#include "kernels.cuh"
#include "surfaceData.cuh"
#include "utils.cuh"

// Auxiliary functions
void producer_function(TIFF **landsat_bands, int producer_id, std::queue<Task> &task_queue, std::mutex &task_queue_mutex, std::condition_variable &task_queue_cv, uint32_t height_band, uint32_t width_band)
{
    for (int band_idx = producer_id; band_idx < 8; band_idx += 4)
    {
        TIFF *tiff = landsat_bands[band_idx];
        unsigned short scanline_size = TIFFScanlineSize(tiff);
        for (int line_idx = 0; line_idx < height_band; ++line_idx)
        {
            tdata_t buf = _TIFFmalloc(scanline_size);
            TIFFReadScanline(tiff, buf, line_idx);

            Task task{band_idx, line_idx, buf, width_band, scanline_size};
            {
                std::unique_lock<std::mutex> lock(task_queue_mutex);
                task_queue.push(std::move(task));
            }
            task_queue_cv.notify_one();
        }
    }
}

void consumer_function(std::queue<Task> &task_queue, std::mutex &task_queue_mutex, std::condition_variable &task_queue_cv, bool &producers_done, float **host_bands)
{
    while (true)
    {
        Task task;
        {
            std::unique_lock<std::mutex> lock(task_queue_mutex);
            task_queue_cv.wait(lock, [&]
                               { return !task_queue.empty() || producers_done; });

            if (task_queue.empty() && producers_done)
                break;

            task = std::move(task_queue.front());
            task_queue.pop();
        }

        unsigned short line_size = task.scanline_size / task.width;
        for (int col = 0; col < task.width; ++col)
        {
            float value = 0;
            memcpy(&value, static_cast<unsigned char *>(task.buffer) + col * line_size, line_size);
            host_bands[task.band_idx][task.line_idx * task.width + col] =
                (task.band_idx == 7) ? (0.75f + 2e-5f * value) : value;
        }

        _TIFFfree(task.buffer);
    }
}

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

    for (int i = 0; i < 8; ++i)
        cudaEventCreate(&copy_done_events[i]);

    HANDLE_ERROR(cudaStreamCreate(&this->stream_1));
    HANDLE_ERROR(cudaStreamCreate(&this->stream_2));
    HANDLE_ERROR(cudaStreamCreate(&this->stream_3));
    HANDLE_ERROR(cudaStreamCreate(&this->stream_4));
    HANDLE_ERROR(cudaStreamCreate(&this->stream_5));
    HANDLE_ERROR(cudaStreamCreate(&this->stream_6));
    HANDLE_ERROR(cudaStreamCreate(&this->stream_7));
    HANDLE_ERROR(cudaStreamCreate(&this->stream_8));

    HANDLE_ERROR(cudaMemcpyToSymbol(width_d, &width_band, sizeof(int), 0, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpyToSymbol(height_d, &height_band, sizeof(int), 0, cudaMemcpyHostToDevice));

    this->stop_condition = (int *)malloc(sizeof(int));
    HANDLE_ERROR(cudaMalloc((void **)&this->stop_condition_d, sizeof(int)));

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
    string result = "";

    float *host_bands[] = {band_blue, band_green, band_red, band_nir, band_swir1, band_termal, band_swir2, tal};
    float *device_bands[] = {band_blue_d, band_green_d, band_red_d, band_nir_d, band_swir1_d, band_termal_d, band_swir2_d, tal_d};
    cudaStream_t streams[] = {stream_1, stream_2, stream_3, stream_4, stream_5, stream_6, stream_7, stream_8};

    // Control structures
    std::queue<Task> task_queue;
    std::mutex task_queue_mutex;
    std::condition_variable task_queue_cv;
    bool producers_done = false;

    begin = system_clock::now();
    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    // Create producer threads
    unsigned int num_producers = 4;
    std::vector<std::thread> producer_threads;
    for (int i = 0; i < num_producers; ++i)
        producer_threads.emplace_back(producer_function, landsat_bands, i, std::ref(task_queue),
                                      std::ref(task_queue_mutex), std::ref(task_queue_cv), height_band, width_band);

    // Determine the number of consumer threads
    unsigned int num_consumers = std::thread::hardware_concurrency() - num_producers;
    if (num_consumers == 0)
        num_consumers = 8;
    std::vector<std::thread> consumer_threads;
    for (int i = 0; i < num_consumers; ++i)
        consumer_threads.emplace_back(consumer_function, std::ref(task_queue), std::ref(task_queue_mutex),
                                      std::ref(task_queue_cv), std::ref(producers_done), host_bands);

    // Wait for producers to finish
    for (auto &t : producer_threads)
        t.join();
    {
        std::unique_lock<std::mutex> lock(task_queue_mutex);
        producers_done = true;
    }
    task_queue_cv.notify_all();

    // Wait for consumers to finish
    for (auto &t : consumer_threads)
        t.join();

    // Transfer data to device
    for (int i = 0; i < 8; ++i)
    {
        HANDLE_ERROR(cudaMemcpyAsync(device_bands[i], host_bands[i], band_bytes, cudaMemcpyHostToDevice, streams[i]));
        cudaEventRecord(copy_done_events[i], streams[i]);
    }

    end = system_clock::now();
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
    general_time = duration_cast<nanoseconds>(end - begin).count() / 1000000.0f;

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

    HANDLE_ERROR(cudaStreamDestroy(stream_1));
    HANDLE_ERROR(cudaStreamDestroy(stream_2));
    HANDLE_ERROR(cudaStreamDestroy(stream_3));
    HANDLE_ERROR(cudaStreamDestroy(stream_4));
    HANDLE_ERROR(cudaStreamDestroy(stream_5));
    HANDLE_ERROR(cudaStreamDestroy(stream_6));
    HANDLE_ERROR(cudaStreamDestroy(stream_7));
    HANDLE_ERROR(cudaStreamDestroy(stream_8));
};
