#include "landsat.h"
#include "cuda_utils.h"
#include <thrust/sort.h>

Landsat::Landsat(string bands_paths[], MTL mtl, int threads_num)
{
  // Open bands
  for (int i = 0; i < 8; i++)
  {
    std::string path_tiff_base = bands_paths[i];
    bands_resampled[i] = TIFFOpen(path_tiff_base.c_str(), "rm");
  }

  // Get bands metadata
  uint16_t sample_format;
  uint32_t height, width;
  TIFFGetField(bands_resampled[1], TIFFTAG_IMAGELENGTH, &height);
  TIFFGetField(bands_resampled[1], TIFFTAG_IMAGEWIDTH, &width);
  TIFFGetField(bands_resampled[1], TIFFTAG_SAMPLEFORMAT, &sample_format);

  this->mtl = mtl;
  this->width_band = width;
  this->height_band = height;
  this->sample_bands = sample_format;
  this->threads_num = threads_num;
  this->products = Products(this->width_band, this->height_band, this->threads_num);

  // Get bands data
  for (int i = 0; i < 7; i++)
  {
    for (int line = 0; line < height; line++)
    {
      TIFF *curr_band = bands_resampled[i];
      tdata_t band_line_buff = _TIFFmalloc(TIFFScanlineSize(curr_band));
      unsigned short curr_band_line_size = TIFFScanlineSize(curr_band) / width;
      TIFFReadScanline(curr_band, band_line_buff, line);

      for (int col = 0; col < width; col++)
      {
        float value = 0;
        memcpy(&value, static_cast<unsigned char *>(band_line_buff) + col * curr_band_line_size, curr_band_line_size);

        switch (i)
        {
        case 0:
          this->products.band_blue[line * width + col] = value;
          break;
        case 1:
          this->products.band_green[line * width + col] = value;
          break;
        case 2:
          this->products.band_red[line * width + col] = value;
          break;
        case 3:
          this->products.band_nir[line * width + col] = value;
          break;
        case 4:
          this->products.band_swir1[line * width + col] = value;
          break;
        case 5:
          this->products.band_termal[line * width + col] = value;
          break;
        case 6:
          this->products.band_swir2[line * width + col] = value;
          break;
        default:
          break;
        }
      }
      _TIFFfree(band_line_buff);
    }
  }

  // Get tal data
  TIFF *elevation_band = this->bands_resampled[7];
  for (int line = 0; line < height; line++)
  {
    tdata_t band_line_buff = _TIFFmalloc(TIFFScanlineSize(elevation_band));
    unsigned short curr_band_line_size = TIFFScanlineSize(elevation_band) / width;
    TIFFReadScanline(elevation_band, band_line_buff, line);

    for (int col = 0; col < width; col++)
    {
      float value = 0;
      memcpy(&value, static_cast<unsigned char *>(band_line_buff) + col * curr_band_line_size, curr_band_line_size);

      this->products.tal[line * width + col] = 0.75 + 2 * pow(10, -5) * value;
    }
    _TIFFfree(band_line_buff);
  }

  HANDLE_ERROR(cudaMemcpy(this->products.band_blue_d, this->products.band_blue, sizeof(float) * height_band * width_band, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(this->products.band_green_d, this->products.band_green, sizeof(float) * height_band * width_band, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(this->products.band_red_d, this->products.band_red, sizeof(float) * height_band * width_band, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(this->products.band_nir_d, this->products.band_nir, sizeof(float) * height_band * width_band, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(this->products.band_swir1_d, this->products.band_swir1, sizeof(float) * height_band * width_band, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(this->products.band_termal_d, this->products.band_termal, sizeof(float) * height_band * width_band, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(this->products.band_swir2_d, this->products.band_swir2, sizeof(float) * height_band * width_band, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(this->products.tal_d, this->products.tal, sizeof(float) * height_band * width_band, cudaMemcpyHostToDevice));
};

string Landsat::compute_Rn_G(Station station)
{
  string result = "";
  int64_t initial_time, final_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  cudaEventRecord(start);
  result += products.radiance_function(mtl);
  result += products.reflectance_function(mtl);
  result += products.albedo_function(mtl);

  // Vegetation indices
  result += products.ndvi_function();
  result += products.pai_function();
  result += products.lai_function();
  result += products.evi_function();

  // Emissivity indices
  result += products.enb_emissivity_function();
  result += products.eo_emissivity_function();
  result += products.ea_emissivity_function();
  result += products.surface_temperature_function(mtl);

  // Radiation waves
  result += products.short_wave_radiation_function(mtl);
  result += products.large_wave_radiation_surface_function();
  result += products.large_wave_radiation_atmosphere_function(station.temperature_image);

  // Main products
  result += products.net_radiation_function();
  result += products.soil_heat_flux_function();
  cudaEventRecord(stop);

  float cuda_time = 0;
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&cuda_time, start, stop);
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  result += "KERNELS,P1_INITIAL_PROD," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
  return result;
}

string Landsat::select_endmembers(int method)
{
  string result = "";
  int64_t initial_time, final_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  cudaEventRecord(start);
  if (method == 0)
  { // STEEP
    result += getEndmembersSTEEP(products.ndvi_d, products.surface_temperature_d, products.albedo_d, products.net_radiation_d, products.soil_heat_d,
                                 products.blocks_num, products.threads_num, products.d_hotCandidates, products.d_coldCandidates, height_band, width_band);
  }
  else if (method == 1)
  { // ASEBAL
    result += getEndmembersASEBAL(products.ndvi_d, products.surface_temperature_d, products.albedo_d, products.net_radiation_d, products.soil_heat_d,
                                  products.blocks_num, products.threads_num, products.d_hotCandidates, products.d_coldCandidates, height_band, width_band);
  }
  cudaEventRecord(stop);

  float cuda_time = 0;
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&cuda_time, start, stop);
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  result += "KERNELS,P2_PIXEL_SEL," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
  return result;
}

string Landsat::converge_rah_cycle(Station station, int method)
{
  string result = "";
  int64_t initial_time, final_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  cudaEventRecord(start);
  float ustar_station = (VON_KARMAN * station.v6) / (log(station.WIND_SPEED / station.SURFACE_ROUGHNESS));
  float u10 = (ustar_station / VON_KARMAN) * log(10 / station.SURFACE_ROUGHNESS);
  float u200 = (ustar_station / VON_KARMAN) * log(200 / station.SURFACE_ROUGHNESS);

  float ndvi_min = thrust::reduce(thrust::device,
                                  products.ndvi_d,
                                  products.ndvi_d + height_band * width_band,
                                  1.0f, // Initial value
                                  thrust::minimum<float>());

  float ndvi_max = thrust::reduce(thrust::device,
                                  products.ndvi_d,
                                  products.ndvi_d + height_band * width_band,
                                  -1.0f, // Initial value
                                  thrust::maximum<float>());

  result += products.d0_fuction();
  result += products.zom_fuction(station.A_ZOM, station.B_ZOM);

  if (method == 0) // STEEP
    result += products.ustar_fuction(u10);
  else // ASEBAL
    result += products.ustar_fuction(u200);

  result += products.kb_function(ndvi_max, ndvi_min);
  result += products.aerodynamic_resistance_fuction();

  if (method == 0) // STEEP
    result += products.rah_correction_function_blocks_STEEP(ndvi_min, ndvi_max);
  else // ASEBAL
    result += products.rah_correction_function_blocks_ASEBAL(ndvi_min, ndvi_max, u200);
  cudaEventRecord(stop);

  float cuda_time = 0;
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&cuda_time, start, stop);
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  result += "KERNELS,P3_RAH," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
  return result;
};

string Landsat::compute_H_ET(Station station)
{
  string result = "";
  int64_t initial_time, final_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  cudaEventRecord(start);
  float dr = (1 / mtl.distance_earth_sun) * (1 / mtl.distance_earth_sun);
  float sigma = 0.409 * sin(((2 * PI / 365) * mtl.julian_day) - 1.39);
  float phi = (PI / 180) * station.latitude;
  float omegas = acos(-tan(phi) * tan(sigma));
  float Ra24h = (((24 * 60 / PI) * GSC * dr) * (omegas * sin(phi) * sin(sigma) + cos(phi) * cos(sigma) * sin(omegas))) * (1000000 / 86400.0);
  float Rs24h = station.INTERNALIZATION_FACTOR * sqrt(station.v7_max - station.v7_min) * Ra24h;

  result += products.sensible_heat_flux_function();
  result += products.latent_heat_flux_function();
  result += products.net_radiation_24h_function(Ra24h, Rs24h);
  result += products.evapotranspiration_fraction_fuction();
  result += products.sensible_heat_flux_24h_fuction();
  result += products.latent_heat_flux_24h_function();
  result += products.evapotranspiration_24h_function(station);
  result += products.evapotranspiration_function();
  cudaEventRecord(stop);

  float cuda_time = 0;
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&cuda_time, start, stop);
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  result += "KERNELS,P4_FINAL_PROD," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
  return result;
};

string Landsat::copy_to_host()
{
  system_clock::time_point begin, end;
  float general_time;
  int64_t initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  HANDLE_ERROR(cudaMemcpy(products.radiance_blue, products.radiance_blue_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(products.radiance_green, products.radiance_green_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(products.radiance_red, products.radiance_red_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(products.radiance_nir, products.radiance_nir_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(products.radiance_swir1, products.radiance_swir1_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(products.radiance_termal, products.radiance_termal_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(products.radiance_swir2, products.radiance_swir2_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(products.reflectance_blue, products.reflectance_blue_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(products.reflectance_green, products.reflectance_green_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(products.reflectance_red, products.reflectance_red_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(products.reflectance_nir, products.reflectance_nir_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(products.reflectance_swir1, products.reflectance_swir1_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(products.reflectance_termal, products.reflectance_termal_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(products.reflectance_swir2, products.reflectance_swir2_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(products.albedo, products.albedo_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(products.ndvi, products.ndvi_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(products.pai, products.pai_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(products.lai, products.lai_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(products.evi, products.evi_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(products.enb_emissivity, products.enb_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(products.eo_emissivity, products.eo_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(products.ea_emissivity, products.ea_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(products.surface_temperature, products.surface_temperature_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(products.short_wave_radiation, products.short_wave_radiation_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(products.large_wave_radiation_surface, products.large_wave_radiation_surface_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(products.large_wave_radiation_atmosphere, products.large_wave_radiation_atmosphere_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(products.net_radiation, products.net_radiation_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(products.soil_heat, products.soil_heat_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(products.d0, products.d0_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(products.zom, products.zom_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(products.ustar, products.ustar_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(products.kb1, products.kb1_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(products.sensible_heat_flux, products.sensible_heat_flux_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(products.latent_heat_flux, products.latent_heat_flux_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(products.net_radiation_24h, products.net_radiation_24h_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(products.evapotranspiration_fraction, products.evapotranspiration_fraction_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(products.sensible_heat_flux_24h, products.sensible_heat_flux_24h_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(products.latent_heat_flux_24h, products.latent_heat_flux_24h_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(products.evapotranspiration_24h, products.evapotranspiration_24h_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(products.evapotranspiration, products.evapotranspiration_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count() / 1000000.0;
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  return "SERIAL,P5_COPY_HOST," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
}

string Landsat::save_products(string output_path)
{
  system_clock::time_point begin, end;
  float general_time;
  int64_t initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  saveTiff(output_path + "/albedo.tif", products.albedo, height_band, width_band);
  saveTiff(output_path + "/ndvi.tif", products.ndvi, height_band, width_band);
  saveTiff(output_path + "/pai.tif", products.pai, height_band, width_band);
  saveTiff(output_path + "/lai.tif", products.lai, height_band, width_band);
  saveTiff(output_path + "/evi.tif", products.evi, height_band, width_band);
  saveTiff(output_path + "/enb_emissivity.tif", products.enb_emissivity, height_band, width_band);
  saveTiff(output_path + "/eo_emissivity.tif", products.eo_emissivity, height_band, width_band);
  saveTiff(output_path + "/ea_emissivity.tif", products.ea_emissivity, height_band, width_band);
  saveTiff(output_path + "/surface_temperature.tif", products.surface_temperature, height_band, width_band);
  saveTiff(output_path + "/net_radiation.tif", products.net_radiation, height_band, width_band);
  saveTiff(output_path + "/soil_heat_flux.tif", products.soil_heat, height_band, width_band);
  saveTiff(output_path + "/d0.tif", products.d0, height_band, width_band);
  saveTiff(output_path + "/zom.tif", products.zom, height_band, width_band);
  saveTiff(output_path + "/ustar.tif", products.ustar, height_band, width_band);
  saveTiff(output_path + "/kb.tif", products.kb1, height_band, width_band);
  saveTiff(output_path + "/rah.tif", products.aerodynamic_resistance, height_band, width_band);
  saveTiff(output_path + "/sensible_heat_flux.tif", products.sensible_heat_flux, height_band, width_band);
  saveTiff(output_path + "/latent_heat_flux.tif", products.latent_heat_flux, height_band, width_band);
  saveTiff(output_path + "/net_radiation_24h.tif", products.net_radiation_24h, height_band, width_band);
  saveTiff(output_path + "/evapotranspiration_fraction.tif", products.evapotranspiration_fraction, height_band, width_band);
  saveTiff(output_path + "/sensible_heat_flux_24h.tif", products.sensible_heat_flux_24h, height_band, width_band);
  saveTiff(output_path + "/latent_heat_flux_24h.tif", products.latent_heat_flux_24h, height_band, width_band);
  saveTiff(output_path + "/evapotranspiration_24h.tif", products.evapotranspiration_24h, height_band, width_band);
  saveTiff(output_path + "/evapotranspiration.tif", products.evapotranspiration, height_band, width_band);

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count() / 1000000.0;
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  return "SERIAL,P6_SAVE_PRODS," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Landsat::print_products(string output_path)
{
  system_clock::time_point begin, end;
  float general_time;
  int64_t initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  // redirect the stdout to a file]
  std::ofstream
      out(output_path + "/products.txt");
  std::streambuf *coutbuf = std::cout.rdbuf();
  std::cout.rdbuf(out.rdbuf());

  std::cout << "==== Albedo" << std::endl;
  printLinearPointer(products.albedo, height_band, width_band);

  std::cout << "==== NDVI" << std::endl;
  printLinearPointer(products.ndvi, height_band, width_band);

  std::cout << "==== PAI" << std::endl;
  printLinearPointer(products.pai, height_band, width_band);

  std::cout << "==== LAI" << std::endl;
  printLinearPointer(products.lai, height_band, width_band);

  std::cout << "==== EVI" << std::endl;
  printLinearPointer(products.evi, height_band, width_band);

  std::cout << "==== ENB Emissivity" << std::endl;
  printLinearPointer(products.enb_emissivity, height_band, width_band);

  std::cout << "==== EO Emissivity" << std::endl;
  printLinearPointer(products.eo_emissivity, height_band, width_band);

  std::cout << "==== EA Emissivity" << std::endl;
  printLinearPointer(products.ea_emissivity, height_band, width_band);

  std::cout << "==== Surface Temperature" << std::endl;
  printLinearPointer(products.surface_temperature, height_band, width_band);

  std::cout << "==== Net Radiation" << std::endl;
  printLinearPointer(products.net_radiation, height_band, width_band);

  std::cout << "==== Soil Heat Flux" << std::endl;
  printLinearPointer(products.soil_heat, height_band, width_band);

  std::cout << "==== D0" << std::endl;
  printLinearPointer(products.d0, height_band, width_band);

  std::cout << "==== ZOM" << std::endl;
  printLinearPointer(products.zom, height_band, width_band);

  std::cout << "==== Ustar" << std::endl;
  printLinearPointer(products.ustar, height_band, width_band);

  std::cout << "==== KB" << std::endl;
  printLinearPointer(products.kb1, height_band, width_band);

  std::cout << "==== RAH" << std::endl;
  printLinearPointer(products.aerodynamic_resistance, height_band, width_band);

  std::cout << "==== Sensible Heat Flux" << std::endl;
  printLinearPointer(products.sensible_heat_flux, height_band, width_band);

  std::cout << "==== Latent Heat Flux" << std::endl;
  printLinearPointer(products.latent_heat_flux, height_band, width_band);

  std::cout << "==== Net Radiation 24h" << std::endl;
  printLinearPointer(products.net_radiation_24h, height_band, width_band);

  std::cout << "==== Evapotranspiration Fraction" << std::endl;
  printLinearPointer(products.evapotranspiration_fraction, height_band, width_band);

  std::cout << "==== Sensible Heat Flux 24h" << std::endl;
  printLinearPointer(products.sensible_heat_flux_24h, height_band, width_band);

  std::cout << "==== Latent Heat Flux 24h" << std::endl;
  printLinearPointer(products.latent_heat_flux_24h, height_band, width_band);

  std::cout << "==== Evapotranspiration 24h" << std::endl;
  printLinearPointer(products.evapotranspiration_24h, height_band, width_band);

  std::cout << "==== Evapotranspiration" << std::endl;
  printLinearPointer(products.evapotranspiration, height_band, width_band);

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count() / 1000000.0;
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  return "SERIAL,P7_STDOUT_PRODS," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

void Landsat::close()
{
  for (int i = 1; i < 8; i++)
  {
    TIFFClose(this->bands_resampled[i]);
  }
};