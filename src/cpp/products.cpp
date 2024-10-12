#include "products.h"
#include "kernels.h"

Products::Products() {}

Products::Products(uint32_t width_band, uint32_t height_band, int threads_num)
{
  this->threads_num = threads_num;
  this->width_band = width_band;
  this->height_band = height_band;
  this->nBytes_band = height_band * width_band * sizeof(float);

  this->band_blue = (float *)malloc(nBytes_band);
  this->band_green = (float *)malloc(nBytes_band);
  this->band_red = (float *)malloc(nBytes_band);
  this->band_nir = (float *)malloc(nBytes_band);
  this->band_swir1 = (float *)malloc(nBytes_band);
  this->band_termal = (float *)malloc(nBytes_band);
  this->band_swir2 = (float *)malloc(nBytes_band);
  this->tal = (float *)malloc(nBytes_band);

  this->radiance_blue = (float *)malloc(nBytes_band);
  this->radiance_green = (float *)malloc(nBytes_band);
  this->radiance_red = (float *)malloc(nBytes_band);
  this->radiance_nir = (float *)malloc(nBytes_band);
  this->radiance_swir1 = (float *)malloc(nBytes_band);
  this->radiance_termal = (float *)malloc(nBytes_band);
  this->radiance_swir2 = (float *)malloc(nBytes_band);

  this->reflectance_blue = (float *)malloc(nBytes_band);
  this->reflectance_green = (float *)malloc(nBytes_band);
  this->reflectance_red = (float *)malloc(nBytes_band);
  this->reflectance_nir = (float *)malloc(nBytes_band);
  this->reflectance_swir1 = (float *)malloc(nBytes_band);
  this->reflectance_termal = (float *)malloc(nBytes_band);
  this->reflectance_swir2 = (float *)malloc(nBytes_band);

  this->albedo = (float *)malloc(nBytes_band);
  this->ndvi = (float *)malloc(nBytes_band);
  this->soil_heat = (float *)malloc(nBytes_band);
  this->surface_temperature = (float *)malloc(nBytes_band);
  this->net_radiation = (float *)malloc(nBytes_band);
  this->lai = (float *)malloc(nBytes_band);
  this->savi = (float *)malloc(nBytes_band);
  this->evi = (float *)malloc(nBytes_band);
  this->pai = (float *)malloc(nBytes_band);
  this->enb_emissivity = (float *)malloc(nBytes_band);
  this->eo_emissivity = (float *)malloc(nBytes_band);
  this->ea_emissivity = (float *)malloc(nBytes_band);
  this->short_wave_radiation = (float *)malloc(nBytes_band);
  this->large_wave_radiation_surface = (float *)malloc(nBytes_band);
  this->large_wave_radiation_atmosphere = (float *)malloc(nBytes_band);

  this->surface_temperature = (float *)malloc(nBytes_band);
  this->d0 = (float *)malloc(nBytes_band);
  this->zom = (float *)malloc(nBytes_band);
  this->ustar = (float *)malloc(nBytes_band);
  this->kb1 = (float *)malloc(nBytes_band);
  this->aerodynamic_resistance = (float *)malloc(nBytes_band);
  this->sensible_heat_flux = (float *)malloc(nBytes_band);

  this->latent_heat_flux = (float *)malloc(nBytes_band);
  this->net_radiation_24h = (float *)malloc(nBytes_band);
  this->evapotranspiration_fraction = (float *)malloc(nBytes_band);
  this->sensible_heat_flux_24h = (float *)malloc(nBytes_band);
  this->latent_heat_flux_24h = (float *)malloc(nBytes_band);
  this->evapotranspiration_24h = (float *)malloc(nBytes_band);
  this->evapotranspiration = (float *)malloc(nBytes_band);
};

void Products::close()
{
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
  free(this->evi);
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
  free(this->evapotranspiration_fraction);
  free(this->sensible_heat_flux_24h);
  free(this->latent_heat_flux_24h);
  free(this->evapotranspiration_24h);
  free(this->evapotranspiration);
};

string Products::radiance_function(MTL mtl)
{
  // https://www.usgs.gov/landsat-missions/using-usgs-landsat-level-1-data-product
  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  for (int i = 0; i < this->height_band * this->width_band; i++)
  {
    this->radiance_blue[i] = this->band_blue[i] * mtl.rad_mult[PARAM_BAND_BLUE_INDEX] + mtl.rad_add[PARAM_BAND_BLUE_INDEX];
    this->radiance_green[i] = this->band_green[i] * mtl.rad_mult[PARAM_BAND_GREEN_INDEX] + mtl.rad_add[PARAM_BAND_GREEN_INDEX];
    this->radiance_red[i] = this->band_red[i] * mtl.rad_mult[PARAM_BAND_RED_INDEX] + mtl.rad_add[PARAM_BAND_RED_INDEX];
    this->radiance_nir[i] = this->band_nir[i] * mtl.rad_mult[PARAM_BAND_NIR_INDEX] + mtl.rad_add[PARAM_BAND_NIR_INDEX];
    this->radiance_swir1[i] = this->band_swir1[i] * mtl.rad_mult[PARAM_BAND_SWIR1_INDEX] + mtl.rad_add[PARAM_BAND_SWIR1_INDEX];
    this->radiance_termal[i] = this->band_termal[i] * mtl.rad_mult[PARAM_BAND_TERMAL_INDEX] + mtl.rad_add[PARAM_BAND_TERMAL_INDEX];
    this->radiance_swir2[i] = this->band_swir2[i] * mtl.rad_mult[PARAM_BAND_SWIR2_INDEX] + mtl.rad_add[PARAM_BAND_SWIR2_INDEX];

    if (radiance_blue[i] <= 0)
      this->radiance_blue[i] = NAN;
    if (radiance_green[i] <= 0)
      this->radiance_green[i] = NAN;
    if (radiance_red[i] <= 0)
      this->radiance_red[i] = NAN;
    if (radiance_nir[i] <= 0)
      this->radiance_nir[i] = NAN;
    if (radiance_swir1[i] <= 0)
      this->radiance_swir1[i] = NAN;
    if (radiance_termal[i] <= 0)
      this->radiance_termal[i] = NAN;
    if (radiance_swir2[i] <= 0)
      this->radiance_swir2[i] = NAN;
  }

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
  return "SERIAL,RADIANCE," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
}

string Products::reflectance_function(MTL mtl)
{
  // https://www.usgs.gov/landsat-missions/using-usgs-landsat-level-1-data-product
  const float sin_sun = sin(mtl.sun_elevation * PI / 180);

  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  for (int i = 0; i < this->height_band * this->width_band; i++)
  {
    this->reflectance_blue[i] = (this->band_blue[i] * mtl.ref_mult[PARAM_BAND_BLUE_INDEX] + mtl.ref_add[PARAM_BAND_BLUE_INDEX]) / sin_sun;
    this->reflectance_green[i] = (this->band_green[i] * mtl.ref_mult[PARAM_BAND_GREEN_INDEX] + mtl.ref_add[PARAM_BAND_GREEN_INDEX]) / sin_sun;
    this->reflectance_red[i] = (this->band_red[i] * mtl.ref_mult[PARAM_BAND_RED_INDEX] + mtl.ref_add[PARAM_BAND_RED_INDEX]) / sin_sun;
    this->reflectance_nir[i] = (this->band_nir[i] * mtl.ref_mult[PARAM_BAND_NIR_INDEX] + mtl.ref_add[PARAM_BAND_NIR_INDEX]) / sin_sun;
    this->reflectance_swir1[i] = (this->band_swir1[i] * mtl.ref_mult[PARAM_BAND_SWIR1_INDEX] + mtl.ref_add[PARAM_BAND_SWIR1_INDEX]) / sin_sun;
    this->reflectance_termal[i] = (this->band_termal[i] * mtl.ref_mult[PARAM_BAND_TERMAL_INDEX] + mtl.ref_add[PARAM_BAND_TERMAL_INDEX]) / sin_sun;
    this->reflectance_swir2[i] = (this->band_swir2[i] * mtl.ref_mult[PARAM_BAND_SWIR2_INDEX] + mtl.ref_add[PARAM_BAND_SWIR2_INDEX]) / sin_sun;

    if (reflectance_blue[i] <= 0)
      this->reflectance_blue[i] = NAN;
    if (reflectance_green[i] <= 0)
      this->reflectance_green[i] = NAN;
    if (reflectance_red[i] <= 0)
      this->reflectance_red[i] = NAN;
    if (reflectance_nir[i] <= 0)
      this->reflectance_nir[i] = NAN;
    if (reflectance_swir1[i] <= 0)
      this->reflectance_swir1[i] = NAN;
    if (reflectance_termal[i] <= 0)
      this->reflectance_termal[i] = NAN;
    if (reflectance_swir2[i] <= 0)
      this->reflectance_swir2[i] = NAN;
  }

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
  return "SERIAL,REFLECTANCE," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
}

string Products::albedo_function(MTL mtl)
{
  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  // https://doi.org/10.1016/j.rse.2017.10.031
  for (int i = 0; i < this->height_band * this->width_band; i++)
  {
    float alb = this->reflectance_blue[i] * mtl.ref_w_coeff[PARAM_BAND_BLUE_INDEX] +
                this->reflectance_green[i] * mtl.ref_w_coeff[PARAM_BAND_GREEN_INDEX] +
                this->reflectance_red[i] * mtl.ref_w_coeff[PARAM_BAND_RED_INDEX] +
                this->reflectance_nir[i] * mtl.ref_w_coeff[PARAM_BAND_NIR_INDEX] +
                this->reflectance_swir1[i] * mtl.ref_w_coeff[PARAM_BAND_SWIR1_INDEX] +
                this->reflectance_swir2[i] * mtl.ref_w_coeff[PARAM_BAND_SWIR2_INDEX];

    this->albedo[i] = (alb - 0.03) / (this->tal[i] * this->tal[i]);

    if (albedo[i] <= 0)
      this->albedo[i] = NAN;
  }

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
  return "SERIAL,ALBEDO," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
}

string Products::ndvi_function()
{
  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  for (int i = 0; i < this->height_band * this->width_band; i++)
  {
    this->ndvi[i] = (this->reflectance_nir[i] - this->reflectance_red[i]) / (this->reflectance_nir[i] + this->reflectance_red[i]);

    if (ndvi[i] <= -1 || ndvi[i] >= 1)
      ndvi[i] = NAN;
  }

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
  return "SERIAL,NDVI," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::pai_function()
{
  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  for (int i = 0; i < this->height_band * this->width_band; i++)
  {
    float pai_value = 10.1 * (this->reflectance_nir[i] - sqrt(this->reflectance_red[i])) + 3.1;

    if (pai_value < 0)
      pai_value = 0;

    this->pai[i] = pai_value;
  }

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
  return "SERIAL,PAI," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::lai_function()
{
  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  for (int i = 0; i < this->height_band * this->width_band; i++)
  {
    float savi = ((1 + 0.5) * (this->reflectance_nir[i] - this->reflectance_red[i])) / (0.5 + (this->reflectance_nir[i] + this->reflectance_red[i]));
    this->savi[i] = savi;

    if (!isnan(savi) && savi > 0.687)
      this->lai[i] = 6;
    if (!isnan(savi) && savi <= 0.687)
      this->lai[i] = -log((0.69 - savi) / 0.59) / 0.91;
    if (!isnan(savi) && savi < 0.1)
      this->lai[i] = 0;

    if (lai[i] < 0)
      lai[i] = 0;
  }

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
  return "SERIAL,LAI," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::evi_function()
{
  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  for (int i = 0; i < this->height_band * this->width_band; i++)
  {
    float evi_value = 2.5 * ((this->reflectance_nir[i] - this->reflectance_red[i]) / (this->reflectance_nir[i] + (6 * this->reflectance_red[i]) - (7.5 * this->reflectance_blue[i]) + 1));

    if (evi_value < 0)
      evi_value = 0;

    this->evi[i] = evi_value;
  }

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
  return "SERIAL,EVI," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::enb_emissivity_function()
{
  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  for (int i = 0; i < this->height_band * this->width_band; i++)
  {
    if (this->lai[i] == 0)
      this->enb_emissivity[i] = NAN;
    else
      this->enb_emissivity[i] = 0.97 + 0.0033 * this->lai[i];

    if ((ndvi[i] < 0) || (lai[i] > 2.99))
      this->enb_emissivity[i] = 0.98;
  }

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
  return "SERIAL,ENB_EMISSIVITY," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::eo_emissivity_function()
{
  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  for (int i = 0; i < this->height_band * this->width_band; i++)
  {

    if (this->lai[i] == 0)
      this->eo_emissivity[i] = NAN;
    else
      this->eo_emissivity[i] = 0.95 + 0.01 * this->lai[i];

    if ((this->ndvi[i] < 0) || (this->lai[i] > 2.99))
      this->eo_emissivity[i] = 0.98;
  }

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
  return "SERIAL,EO_EMISSIVITY," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::ea_emissivity_function()
{
  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  for (int i = 0; i < this->height_band * this->width_band; i++)
    this->ea_emissivity[i] = 0.85 * pow((-1 * log(this->tal[i])), 0.09);

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
  return "SERIAL,EA_EMISSIVITY," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::surface_temperature_function(MTL mtl)
{
  float k1, k2;
  switch (mtl.number_sensor)
  {
  case 5:
    k1 = 607.76;
    k2 = 1260.56;
    break;

  case 7:
    k1 = 666.09;
    k2 = 1282.71;
    break;

  case 8:
    k1 = 774.8853;
    k2 = 1321.0789;
    break;

  default:
    cerr << "Sensor problem!";
    exit(6);
  }

  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  float surface_temperature_value;
  for (int i = 0; i < this->height_band * this->width_band; i++)
  {
    surface_temperature_value = k2 / (log((this->enb_emissivity[i] * k1 / this->radiance_termal[i]) + 1));

    if (surface_temperature_value < 0)
      surface_temperature_value = 0;

    this->surface_temperature[i] = surface_temperature_value;
  }

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
  return "SERIAL,SURFACE_TEMPERATURE," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::short_wave_radiation_function(MTL mtl)
{
  float costheta = sin(mtl.sun_elevation * PI / 180);

  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  for (int i = 0; i < this->height_band * this->width_band; i++)
    this->short_wave_radiation[i] = (1367 * costheta * this->tal[i]) / (mtl.distance_earth_sun * mtl.distance_earth_sun);

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
  return "SERIAL,SHORT_WAVE_RADIATION," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::large_wave_radiation_surface_function()
{
  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  for (int i = 0; i < this->height_band * this->width_band; i++)
  {
    float temperature_pixel = this->surface_temperature[i];
    float surface_temperature_pow_4 = temperature_pixel * temperature_pixel * temperature_pixel * temperature_pixel;
    this->large_wave_radiation_surface[i] = this->eo_emissivity[i] * 5.67 * 1e-8 * surface_temperature_pow_4;
  }

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
  return "SERIAL,LARGE_WAVE_RADIATION_SURFACE," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::large_wave_radiation_atmosphere_function(float temperature)
{
  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  float temperature_kelvin = temperature + 273.15;
  float temperature_kelvin_pow_4 = temperature_kelvin * temperature_kelvin * temperature_kelvin * temperature_kelvin;

  for (int i = 0; i < this->height_band * this->width_band; i++)
    this->large_wave_radiation_atmosphere[i] = this->ea_emissivity[i] * 5.67 * 1e-8 * temperature_kelvin_pow_4;

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
  return "SERIAL,LARGE_WAVE_RADIATION_ATMOSPHERE," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::net_radiation_function()
{
  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  for (int i = 0; i < this->height_band * this->width_band; i++)
  {
    this->net_radiation[i] = this->short_wave_radiation[i] -
                             (this->short_wave_radiation[i] * this->albedo[i]) +
                             this->large_wave_radiation_atmosphere[i] - this->large_wave_radiation_surface[i] -
                             (1 - this->eo_emissivity[i]) * this->large_wave_radiation_atmosphere[i];

    if (this->net_radiation[i] < 0)
      this->net_radiation[i] = 0;
  }

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
  return "SERIAL,NET_RADIATION," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::soil_heat_flux_function()
{
  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  for (int i = 0; i < this->height_band * this->width_band; i++)
  {
    if ((this->ndvi[i] < 0) || this->ndvi[i] > 0)
    {
      float ndvi_pixel_pow_4 = this->ndvi[i] * this->ndvi[i] * this->ndvi[i] * this->ndvi[i];
      this->soil_heat[i] = (this->surface_temperature[i] - 273.15) * (0.0038 + 0.0074 * this->albedo[i]) *
                           (1 - 0.98 * ndvi_pixel_pow_4) * this->net_radiation[i];
    }
    else
      this->soil_heat[i] = 0.5 * this->net_radiation[i];

    if (this->soil_heat[i] < 0)
      this->soil_heat[i] = 0;
  }

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
  return "SERIAL,SOIL_HEAT_FLUX," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::d0_fuction()
{
  float CD1 = 20.6;
  float HGHT = 4;

  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  for (int i = 0; i < this->height_band * this->width_band; i++)
  {
    float cd1_pai_root = sqrt(CD1 * this->pai[i]);
    this->d0[i] = HGHT * ((1 - (1 / cd1_pai_root)) + (pow(exp(1.0), -cd1_pai_root) / cd1_pai_root));
  }

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
  return "SERIAL,D0," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::kb_function(float ndvi_max, float ndvi_min)
{
  float HGHT = 4;

  float visc = 0.00001461;
  float pr = 0.71;
  float c1 = 0.320;
  float c2 = 0.264;
  float c3 = 15.1;
  float cd = 0.2;
  float ct = 0.01;
  float sf_c = 0.3;
  float sf_d = 2.5;
  float sf_e = 4.0;

  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  for (int i = 0; i < this->height_band * this->width_band; i++)
  {
    float zom = this->zom[i];
    float u_ast_ini_terra = this->ustar[i];
    float PAI = this->pai[i];

    float Re_star = (u_ast_ini_terra * 0.009) / visc;
    float Ct_star = pow(pr, -0.667) * pow(Re_star, -0.5);
    float beta = c1 - c2 * (exp((cd * -c3 * PAI)));
    float nec_terra = (cd * PAI) / (beta * beta * 2);

    float kb1_fst_part = (cd * VON_KARMAN) / (4 * ct * beta * (1 - exp(nec_terra * -0.5)));
    float kb1_sec_part = (beta * VON_KARMAN * (zom / HGHT)) / Ct_star;
    float kb1s = (pow(Re_star, 0.25) * 2.46) - 2;

    float fc = 1 - pow((this->ndvi[i] - ndvi_max) / (ndvi_min - ndvi_max), 0.4631);
    float fs = 1 - fc;

    float soil_moisture_day_rel = 0.33;

    float SF = sf_c + (1 / (1 + pow(exp(1.0), (sf_d - (sf_e * soil_moisture_day_rel)))));

    this->kb1[i] = ((kb1_fst_part * pow(fc, 2)) +
                    (kb1_sec_part * pow(fc, 2) * pow(fs, 2)) +
                    (pow(fs, 2) * kb1s)) *
                   SF;
  }

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
  return "SERIAL,KB1," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::zom_fuction(float A_ZOM, float B_ZOM)
{
  float HGHT = 4;
  float CD = 0.01;
  float CR = 0.35;
  float PSICORR = 0.2;

  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  float gama;
  for (int i = 0; i < this->height_band * this->width_band; i++)
  {
    gama = pow((CD + CR * (this->pai[i] / 2)), -0.5);

    if (gama < 3.3)
      gama = 3.3;

    this->zom[i] = (HGHT - this->d0[i]) * pow(exp(1.0), (-VON_KARMAN * gama) + PSICORR);
  }

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
  return "SERIAL,ZOM," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::ustar_fuction(float u10)
{
  float zu = 10;

  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  for (int i = 0; i < this->height_band * this->width_band; i++)
  {
    float DISP = this->d0[i];
    float zom = this->zom[i];
    this->ustar[i] = (u10 * VON_KARMAN) / log((zu - DISP) / zom);
  }

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
  return "SERIAL,USTAR," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::aerodynamic_resistance_fuction()
{
  float zu = 10.0;

  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  for (int i = 0; i < this->height_band * this->width_band; i++)
  {
    float DISP = this->d0[i];
    float zom = this->zom[i];
    float zoh_terra = zom / pow(exp(1.0), (this->kb1[i]));

    float temp_kb_1_terra = log(zom / zoh_terra);
    float temp_rah1_terra = (1 / (this->ustar[i] * VON_KARMAN));
    float temp_rah2 = log(((zu - DISP) / zom));
    float temp_rah3_terra = temp_rah1_terra * temp_kb_1_terra;

    this->aerodynamic_resistance[i] = temp_rah1_terra * temp_rah2 + temp_rah3_terra;
  }

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
  return "SERIAL,RAH_INI," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::sensible_heat_flux_function(float a, float b)
{
  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  for (int i = 0; i < this->height_band * this->width_band; i++)
  {
    this->sensible_heat_flux[i] = RHO * SPECIFIC_HEAT_AIR * (a + b * (this->surface_temperature[i] - 273.15)) / this->aerodynamic_resistance[i];

    if (!isnan(this->sensible_heat_flux[i]) && this->sensible_heat_flux[i] > (this->net_radiation[i] - this->soil_heat[i]))
    {
      this->sensible_heat_flux[i] = this->net_radiation[i] - this->soil_heat[i];
    }
  }

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
  return "SERIAL,SENSIBLE_HEAT_FLUX," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::latent_heat_flux_function()
{
  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  for (int i = 0; i < this->height_band * this->width_band; i++)
    this->latent_heat_flux[i] = this->net_radiation[i] - this->soil_heat[i] - this->sensible_heat_flux[i];

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
  return "SERIAL,LATENT_HEAT_FLUX," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::net_radiation_24h_function(float Ra24h, float Rs24h)
{
  int FL = 110;

  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  for (int i = 0; i < this->height_band * this->width_band; i++)
    this->net_radiation_24h[i] = (1 - this->albedo[i]) * Rs24h - FL * Rs24h / Ra24h;

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
  return "SERIAL,NET_RADIATION_24H," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::evapotranspiration_fraction_fuction()
{
  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  for (int i = 0; i < this->height_band * this->width_band; i++)
    this->evapotranspiration_fraction[i] = this->latent_heat_flux[i] / (this->net_radiation[i] - this->soil_heat[i]);

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
  return "SERIAL,EVAPOTRANSPIRATION_FRACTION," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::sensible_heat_flux_24h_fuction()
{
  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  for (int i = 0; i < this->height_band * this->width_band; i++)
    this->sensible_heat_flux_24h[i] = (1 - this->evapotranspiration_fraction[i]) * this->net_radiation_24h[i];

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
  return "SERIAL,SENSIBLE_HEAT_FLUX_24H," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::latent_heat_flux_24h_function()
{
  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  for (int i = 0; i < this->height_band * this->width_band; i++)
    this->latent_heat_flux_24h[i] = this->evapotranspiration_fraction[i] * this->net_radiation_24h[i];

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
  return "SERIAL,LATENT_HEAT_FLUX_24H," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::evapotranspiration_24h_function(Station station)
{
  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  for (int i = 0; i < this->height_band * this->width_band; i++)
    this->evapotranspiration_24h[i] = (this->latent_heat_flux_24h[i] * 86400) / ((2.501 - 0.00236 * (station.v7_max + station.v7_min) / 2) * 1e+6);

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
  return "SERIAL,EVAPOTRANSPIRATION_24H," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::evapotranspiration_function()
{
  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  for (int i = 0; i < this->height_band * this->width_band; i++)
    this->evapotranspiration[i] = this->net_radiation_24h[i] * this->evapotranspiration_fraction[i] * 0.035;

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
  return "SERIAL,EVAPOTRANSPIRATION," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::rah_correction_function_serial_ASEBAL(float ndvi_min, float ndvi_max, Candidate hot_pixel, Candidate cold_pixel, float u200)
{
  system_clock::time_point begin_core, end_core;
  int64_t general_time_core, initial_time_core, final_time_core;

  float hot_pixel_aerodynamic = aerodynamic_resistance[hot_pixel.line * width_band + hot_pixel.col];
  hot_pixel.setAerodynamicResistance(hot_pixel_aerodynamic);

  float cold_pixel_aerodynamic = aerodynamic_resistance[cold_pixel.line * width_band + cold_pixel.col];
  cold_pixel.setAerodynamicResistance(cold_pixel_aerodynamic);

  float fc_hot = 1 - pow((ndvi[hot_pixel.line * width_band + hot_pixel.col] - ndvi_max) / (ndvi_min - ndvi_max), 0.4631);
  float fc_cold = 1 - pow((ndvi[cold_pixel.line * width_band + cold_pixel.col] - ndvi_max) / (ndvi_min - ndvi_max), 0.4631);

  int i = 0;
  while (true) 
  {      
    this->rah_ini_pq_terra = hot_pixel.aerodynamic_resistance;
    this->rah_ini_pf_terra = cold_pixel.aerodynamic_resistance;

    float LEc_terra = 0.55 * fc_hot * (hot_pixel.net_radiation - hot_pixel.soil_heat_flux) * 0.78;
    float LEc_terra_pf = 1.75 * fc_cold * (cold_pixel.net_radiation - cold_pixel.soil_heat_flux) * 0.78;

    this->H_pf_terra = cold_pixel.net_radiation - cold_pixel.soil_heat_flux - LEc_terra_pf;
    float dt_pf_terra = H_pf_terra * rah_ini_pf_terra / (RHO * SPECIFIC_HEAT_AIR);

    this->H_pq_terra = hot_pixel.net_radiation - hot_pixel.soil_heat_flux - LEc_terra;
    float dt_pq_terra = H_pq_terra * rah_ini_pq_terra / (RHO * SPECIFIC_HEAT_AIR);

    float b = (dt_pq_terra - dt_pf_terra) / (hot_pixel.temperature - cold_pixel.temperature);
    float a = dt_pf_terra - (b * (cold_pixel.temperature - 273.15));

    // ==== Paralelization core
    begin_core = system_clock::now();
    initial_time_core = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    rah_correction_kernel_ASEBAL(0, height_band, width_band, a, b, surface_temperature,
                               zom, kb1, sensible_heat_flux, ustar, u200,
                               aerodynamic_resistance);

    end_core = system_clock::now();
    general_time_core = duration_cast<nanoseconds>(end_core - begin_core).count();
    final_time_core = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
    // ==== Paralelization core

    float rah_hot = this->aerodynamic_resistance[hot_pixel.line * width_band + hot_pixel.col];
    hot_pixel.setAerodynamicResistance(rah_hot);

    float rah_cold = this->aerodynamic_resistance[cold_pixel.line * width_band + cold_pixel.col];
    cold_pixel.setAerodynamicResistance(rah_cold);

    if (i > 0 && fabs(1 - rah_ini_pq_terra / rah_hot) < 0.05)
      break;
    else 
      i++;
  }

  return "SERIAL,RAH_CYCLE," + std::to_string(general_time_core) + "," + std::to_string(initial_time_core) + "," + std::to_string(final_time_core) + "\n";
}

string Products::rah_correction_function_serial_STEEP(float ndvi_min, float ndvi_max, Candidate hot_pixel, Candidate cold_pixel)
{
  system_clock::time_point begin_core, end_core;
  int64_t general_time_core, initial_time_core, final_time_core;

  // ==== Threads setup
  int lines_per_thread = height_band / threads_num;
  thread threads[threads_num];
  // ==== Threads setup

  float hot_pixel_aerodynamic = aerodynamic_resistance[hot_pixel.line * width_band + hot_pixel.col];
  hot_pixel.setAerodynamicResistance(hot_pixel_aerodynamic);

  float cold_pixel_aerodynamic = aerodynamic_resistance[cold_pixel.line * width_band + cold_pixel.col];
  cold_pixel.setAerodynamicResistance(cold_pixel_aerodynamic);

  float fc_hot = 1 - pow((ndvi[hot_pixel.line * width_band + hot_pixel.col] - ndvi_max) / (ndvi_min - ndvi_max), 0.4631);
  float fc_cold = 1 - pow((ndvi[cold_pixel.line * width_band + cold_pixel.col] - ndvi_max) / (ndvi_min - ndvi_max), 0.4631);

  for (int i = 0; i < 2; i++)
  {
    this->rah_ini_pq_terra = hot_pixel.aerodynamic_resistance;
    this->rah_ini_pf_terra = cold_pixel.aerodynamic_resistance;

    float LEc_terra = 0.55 * fc_hot * (hot_pixel.net_radiation - hot_pixel.soil_heat_flux) * 0.78;
    float LEc_terra_pf = 1.75 * fc_cold * (cold_pixel.net_radiation - cold_pixel.soil_heat_flux) * 0.78;

    this->H_pf_terra = cold_pixel.net_radiation - cold_pixel.soil_heat_flux - LEc_terra_pf;
    float dt_pf_terra = H_pf_terra * rah_ini_pf_terra / (RHO * SPECIFIC_HEAT_AIR);

    this->H_pq_terra = hot_pixel.net_radiation - hot_pixel.soil_heat_flux - LEc_terra;
    float dt_pq_terra = H_pq_terra * rah_ini_pq_terra / (RHO * SPECIFIC_HEAT_AIR);

    float b = (dt_pq_terra - dt_pf_terra) / (hot_pixel.temperature - cold_pixel.temperature);
    float a = dt_pf_terra - (b * (cold_pixel.temperature - 273.15));

    // ==== Paralelization core
    begin_core = system_clock::now();
    initial_time_core = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    rah_correction_kernel_STEEP(0, height_band, width_band, a, b, surface_temperature,
                               d0, zom, kb1, sensible_heat_flux, ustar,
                               aerodynamic_resistance);

    end_core = system_clock::now();
    general_time_core = duration_cast<nanoseconds>(end_core - begin_core).count();
    final_time_core = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
    // ==== Paralelization core

    float rah_hot = this->aerodynamic_resistance[hot_pixel.line * width_band + hot_pixel.col];
    hot_pixel.setAerodynamicResistance(rah_hot);

    float rah_cold = this->aerodynamic_resistance[cold_pixel.line * width_band + cold_pixel.col];
    cold_pixel.setAerodynamicResistance(rah_cold);
  }

  return "SERIAL,RAH_CYCLE," + std::to_string(general_time_core) + "," + std::to_string(initial_time_core) + "," + std::to_string(final_time_core) + "\n";
}
