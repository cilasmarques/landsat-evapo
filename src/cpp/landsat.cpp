#include "landsat.h"

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
};

string Landsat::compute_Rn_G(Station station)
{
  string result = "";
  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

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

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
  result += "SERIAL,P1_INITIAL_PROD," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
  return result;
}

string Landsat::select_endmembers(int method)
{
  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  if (method == 0)
  { // STEEP
    pair<Candidate, Candidate> pixels = getEndmembersSTEPP(products.ndvi, products.surface_temperature, products.albedo, products.net_radiation, products.soil_heat, height_band, width_band);
    hot_pixel = pixels.first;
    cold_pixel = pixels.second;
  }
  else if (method == 1)
  { // ASEBAL
    pair<Candidate, Candidate> pixels = getEndmembersASEBAL(products.ndvi, products.surface_temperature, products.albedo, products.net_radiation, products.soil_heat, height_band, width_band);
    hot_pixel = pixels.first;
    cold_pixel = pixels.second;
  }

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  return "SERIAL,P2_PIXEL_SEL," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
}

string Landsat::converge_rah_cycle(Station station, int method)
{
  string result = "";
  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  float ustar_station = (VON_KARMAN * station.v6) / (log(station.WIND_SPEED / station.SURFACE_ROUGHNESS));
  float u10 = (ustar_station / VON_KARMAN) * log(10 / station.SURFACE_ROUGHNESS);
  float ndvi_min = 1.0;
  float ndvi_max = -1.0;

  for (int i = 0; i < this->height_band * this->width_band; i++)
  {
    if (products.ndvi[i] < ndvi_min)
      ndvi_min = products.ndvi[i];
    if (products.ndvi[i] > ndvi_max)
      ndvi_max = products.ndvi[i];
  }

  result += products.d0_fuction();
  result += products.zom_fuction(station.A_ZOM, station.B_ZOM);
  result += products.ustar_fuction(u10);
  result += products.kb_function(ndvi_max, ndvi_min);
  result += products.aerodynamic_resistance_fuction();

  if (threads_num <= 1)
    result += products.rah_correction_function_serial(ndvi_min, ndvi_max, hot_pixel, cold_pixel);
  else
    result += products.rah_correction_function_threads(ndvi_min, ndvi_max, hot_pixel, cold_pixel);

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  result += "SERIAL,P3_RAH," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
  return result;
};

string Landsat::compute_H_ET(Station station)
{
  string result = "";
  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  float dr = (1 / mtl.distance_earth_sun) * (1 / mtl.distance_earth_sun);
  float sigma = 0.409 * sin(((2 * PI / 365) * mtl.julian_day) - 1.39);
  float phi = (PI / 180) * station.latitude;
  float omegas = acos(-tan(phi) * tan(sigma));
  float Ra24h = (((24 * 60 / PI) * GSC * dr) * (omegas * sin(phi) * sin(sigma) + cos(phi) * cos(sigma) * sin(omegas))) * (1000000 / 86400.0);
  float Rs24h = station.INTERNALIZATION_FACTOR * sqrt(station.v7_max - station.v7_min) * Ra24h;

  float dt_pq_terra = products.H_pq_terra * products.rah_ini_pq_terra / (RHO * SPECIFIC_HEAT_AIR);
  float dt_pf_terra = products.H_pf_terra * products.rah_ini_pf_terra / (RHO * SPECIFIC_HEAT_AIR);

  float b = (dt_pq_terra - dt_pf_terra) / (hot_pixel.temperature - cold_pixel.temperature);
  float a = dt_pf_terra - (b * (cold_pixel.temperature - 273.15));

  result += products.sensible_heat_flux_function(a, b);
  result += products.latent_heat_flux_function();
  result += products.net_radiation_24h_function(Ra24h, Rs24h);
  result += products.evapotranspiration_fraction_fuction();
  result += products.sensible_heat_flux_24h_fuction();
  result += products.latent_heat_flux_24h_function();
  result += products.evapotranspiration_24h_function(station);
  result += products.evapotranspiration_function();

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
  result += "SERIAL,P4_FINAL_PROD," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
  return result;
};

void Landsat::save_products(string output_path)
{
  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  std::ofstream outputProds(output_path);
  std::streambuf *coutProds = std::cout.rdbuf();
  std::cout.rdbuf(outputProds.rdbuf());

  std::cout << "==== reflectance 1" << std::endl;
  printLinearPointer(products.reflectance_blue, height_band, width_band);

  std::cout << "==== reflectance 2" << std::endl;
  printLinearPointer(products.reflectance_green, height_band, width_band);

  std::cout << "==== reflectance 3" << std::endl;
  printLinearPointer(products.reflectance_red, height_band, width_band);

  std::cout << "==== reflectance 4" << std::endl;
  printLinearPointer(products.reflectance_nir, height_band, width_band);

  std::cout << "==== reflectance 5" << std::endl;
  printLinearPointer(products.reflectance_swir1, height_band, width_band);

  std::cout << "==== reflectance 6" << std::endl;
  printLinearPointer(products.reflectance_termal, height_band, width_band);

  std::cout << "==== reflectance 7" << std::endl;
  printLinearPointer(products.reflectance_swir2, height_band, width_band);

  std::cout << "==== albedo" << std::endl;
  printLinearPointer(products.albedo, height_band, width_band);

  std::cout << "==== ndvi" << std::endl;
  printLinearPointer(products.ndvi, height_band, width_band);

  std::cout << "==== enb" << std::endl;
  printLinearPointer(products.enb_emissivity, height_band, width_band);

  std::cout << "==== pai" << std::endl;
  printLinearPointer(products.pai, height_band, width_band);

  std::cout << "==== lai" << std::endl;
  printLinearPointer(products.lai, height_band, width_band);

  std::cout << "==== savi" << std::endl;
  printLinearPointer(products.savi, height_band, width_band);

  std::cout << "==== evi" << std::endl;
  printLinearPointer(products.evi, height_band, width_band);

  std::cout << "==== surface_temperature" << std::endl;
  printLinearPointer(products.surface_temperature, height_band, width_band);

  std::cout << "==== short_wave_radiation" << std::endl;
  printLinearPointer(products.short_wave_radiation, height_band, width_band);

  std::cout << "==== large_wave_radiation_surface" << std::endl;
  printLinearPointer(products.large_wave_radiation_surface, height_band, width_band);

  std::cout << "==== large_wave_radiation_atmosphere" << std::endl;
  printLinearPointer(products.large_wave_radiation_atmosphere, height_band, width_band);

  std::cout << "==== ea" << std::endl;
  printLinearPointer(products.ea_emissivity, height_band, width_band);

  std::cout << "==== eo" << std::endl;
  printLinearPointer(products.eo_emissivity, height_band, width_band);

  std::cout << "==== net_radiation" << std::endl;
  printLinearPointer(products.net_radiation, height_band, width_band);

  std::cout << "==== soil_heat" << std::endl;
  printLinearPointer(products.soil_heat, height_band, width_band);

  std::cout << "==== d0" << std::endl;
  printLinearPointer(products.d0, height_band, width_band);

  std::cout << "==== zom" << std::endl;
  printLinearPointer(products.zom, height_band, width_band);

  std::cout << "==== kb1" << std::endl;
  printLinearPointer(products.kb1, height_band, width_band);

  std::cout << "==== ustar" << std::endl;
  printLinearPointer(products.ustar, height_band, width_band);

  std::cout << "==== sensible_heat_flux" << std::endl;
  printLinearPointer(products.sensible_heat_flux, height_band, width_band);

  std::cout << "==== latent_heat_flux" << std::endl;
  printLinearPointer(products.latent_heat_flux, height_band, width_band);

  std::cout << "==== net_radiation_24h" << std::endl;
  printLinearPointer(products.net_radiation_24h, height_band, width_band);

  std::cout << "==== evapotranspiration_fraction" << std::endl;
  printLinearPointer(products.evapotranspiration_fraction, height_band, width_band);

  std::cout << "==== sensible_heat_flux_24h" << std::endl;
  printLinearPointer(products.sensible_heat_flux_24h, height_band, width_band);

  std::cout << "==== latent_heat_flux_24h" << std::endl;
  printLinearPointer(products.latent_heat_flux_24h, height_band, width_band);

  std::cout << "==== evapotranspiration_24h" << std::endl;
  printLinearPointer(products.evapotranspiration_24h, height_band, width_band);

  std::cout << "==== evapotranspiration" << std::endl;
  printLinearPointer(products.evapotranspiration, height_band, width_band);

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
  std::cout << "P3 - WRITE PRODUCTS," << general_time << "," << initial_time << "," << final_time << std::endl;
};

void Landsat::close()
{
  for (int i = 1; i <= 8; i++)
  {
    TIFFClose(this->bands_resampled[i]);
  }
};