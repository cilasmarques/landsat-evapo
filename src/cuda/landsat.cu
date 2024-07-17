#include "landsat.h"
#include "cuda_utils.h"

Landsat::Landsat(string bands_paths[], string land_cover_path, MTL mtl, int threads_num)
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
          this->products.band1[line * width + col] = value;
          break;
        case 1:
          this->products.band2[line * width + col] = value;
          break;
        case 2:
          this->products.band3[line * width + col] = value;
          break;
        case 3:
          this->products.band4[line * width + col] = value;
          break;
        case 4:
          this->products.band5[line * width + col] = value;
          break;
        case 5:
          this->products.band6[line * width + col] = value;
          break;
        case 6:
          this->products.band7[line * width + col] = value;
          break;
        case 7:
          this->products.band8[line * width + col] = value;
          break;
        default:
          break;
        }
      }
      _TIFFfree(band_line_buff);
    }
  }

  // Get tal data
  TIFF *tal_band = this->bands_resampled[7];
  for (int line = 0; line < height; line++)
  {
    tdata_t band_line_buff = _TIFFmalloc(TIFFScanlineSize(tal_band));
    unsigned short curr_band_line_size = TIFFScanlineSize(tal_band) / width;
    TIFFReadScanline(tal_band, band_line_buff, line);

    for (int col = 0; col < width; col++)
    {
      float value = 0;
      memcpy(&value, static_cast<unsigned char *>(band_line_buff) + col * curr_band_line_size, curr_band_line_size);

      this->products.tal[line * width + col] = value;
    }
    _TIFFfree(band_line_buff);
  }

  HANDLE_ERROR(cudaMemcpy(this->products.band1_d, this->products.band1, sizeof(float) * height_band * width_band, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(this->products.band2_d, this->products.band2, sizeof(float) * height_band * width_band, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(this->products.band3_d, this->products.band3, sizeof(float) * height_band * width_band, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(this->products.band4_d, this->products.band4, sizeof(float) * height_band * width_band, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(this->products.band5_d, this->products.band5, sizeof(float) * height_band * width_band, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(this->products.band6_d, this->products.band6, sizeof(float) * height_band * width_band, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(this->products.band7_d, this->products.band7, sizeof(float) * height_band * width_band, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(this->products.band8_d, this->products.band8, sizeof(float) * height_band * width_band, cudaMemcpyHostToDevice));
};

string Landsat::compute_Rn_G(Sensor sensor, Station station)
{
  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  products.radiance_function(mtl, sensor);
  products.reflectance_function(mtl, sensor);
  products.albedo_function(mtl, sensor);

  // Vegetation indices
  products.ndvi_function();
  products.pai_function();
  products.lai_function();
  products.evi_function();

  // Emissivity indices
  products.enb_emissivity_function();
  products.eo_emissivity_function();
  products.ea_emissivity_function();
  products.surface_temperature_function(mtl);

  // Radiation waves
  products.short_wave_radiation_function(mtl);
  products.large_wave_radiation_surface_function();
  products.large_wave_radiation_atmosphere_function(station.temperature_image);

  // Main products
  products.net_radiation_function();
  products.soil_heat_flux_function();

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
  return "P1 - Rn_G," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
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

  return "P2 - PIXEL SELECTION," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
}

string Landsat::converge_rah_cycle(Station station, int method)
{
  string result = "";
  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  double ustar_station = (VON_KARMAN * station.v6) / (log(station.WIND_SPEED / station.SURFACE_ROUGHNESS));
  double u10 = (ustar_station / VON_KARMAN) * log(10 / station.SURFACE_ROUGHNESS);
  double ndvi_min = 1.0;
  double ndvi_max = -1.0;

  for (int i = 0; i < this->height_band * this->width_band; i++)
  {
    if (products.ndvi[i] < ndvi_min)
      ndvi_min = products.ndvi[i];
    if (products.ndvi[i] > ndvi_max)
      ndvi_max = products.ndvi[i];
  }

  products.d0_fuction();
  products.zom_fuction(station.A_ZOM, station.B_ZOM);
  products.ustar_fuction(u10);
  products.kb_function(ndvi_max, ndvi_min);
  products.aerodynamic_resistance_fuction();

  result += products.rah_correction_function_blocks(ndvi_min, ndvi_max, hot_pixel, cold_pixel);

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  result += "P2 - RAH CYCLE," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
  return result;
};

string Landsat::compute_H_ET(Station station)
{
  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  double dr = (1 / mtl.distance_earth_sun) * (1 / mtl.distance_earth_sun);
  double sigma = 0.409 * sin(((2 * PI / 365) * mtl.julian_day) - 1.39);
  double phi = (PI / 180) * station.latitude;
  double omegas = acos(-tan(phi) * tan(sigma));
  double Ra24h = (((24 * 60 / PI) * GSC * dr) * (omegas * sin(phi) * sin(sigma) + cos(phi) * cos(sigma) * sin(omegas))) * (1000000 / 86400.0);
  double Rs24h = station.INTERNALIZATION_FACTOR * sqrt(station.v7_max - station.v7_min) * Ra24h;

  double dt_pq_terra = products.H_pq_terra * products.rah_ini_pq_terra / (RHO * SPECIFIC_HEAT_AIR);
  double dt_pf_terra = products.H_pf_terra * products.rah_ini_pf_terra / (RHO * SPECIFIC_HEAT_AIR);

  double b = (dt_pq_terra - dt_pf_terra) / (hot_pixel.temperature - cold_pixel.temperature);
  double a = dt_pf_terra - (b * (cold_pixel.temperature - 273.15));

  for (int line = 0; line < height_band; line++)
  {
    products.sensible_heat_flux_function(a, b);
    products.latent_heat_flux_function();
    products.net_radiation_24h_function(Ra24h, Rs24h);
    products.evapotranspiration_fraction_fuction();
    products.sensible_heat_flux_24h_fuction();
    products.latent_heat_flux_24h_function();
    products.evapotranspiration_24h_function(station);
    products.evapotranspiration_function();
  }
  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
  return "P2 - FINAL PRODUCTS," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
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

  std::cout << "==== albedo" << std::endl;
  printLinearPointer(products.albedo, height_band, width_band);

  std::cout << "==== ndvi" << std::endl;
  printLinearPointer(products.ndvi, height_band, width_band);

  std::cout << "==== net_radiation" << std::endl;
  printLinearPointer(products.net_radiation, height_band, width_band);

  std::cout << "==== soil_heat" << std::endl;
  printLinearPointer(products.soil_heat, height_band, width_band);

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