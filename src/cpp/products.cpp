#include "products.h"
#include "kernels.h"

Products::Products() {}

Products::Products(uint32_t width_band, uint32_t height_band, int threads_num)
{
  this->threads_num = threads_num;
  this->width_band = width_band;
  this->height_band = height_band;
  this->nBytes_band = height_band * width_band * sizeof(float);

  this->band1 = (float *)malloc(nBytes_band);
  this->band2 = (float *)malloc(nBytes_band);
  this->band3 = (float *)malloc(nBytes_band);
  this->band4 = (float *)malloc(nBytes_band);
  this->band5 = (float *)malloc(nBytes_band);
  this->band6 = (float *)malloc(nBytes_band);
  this->band7 = (float *)malloc(nBytes_band);
  this->tal = (float *)malloc(nBytes_band);

  this->radiance1 = (float *)malloc(nBytes_band);
  this->radiance2 = (float *)malloc(nBytes_band);
  this->radiance3 = (float *)malloc(nBytes_band);
  this->radiance4 = (float *)malloc(nBytes_band);
  this->radiance5 = (float *)malloc(nBytes_band);
  this->radiance6 = (float *)malloc(nBytes_band);
  this->radiance7 = (float *)malloc(nBytes_band);

  this->reflectance1 = (float *)malloc(nBytes_band);
  this->reflectance2 = (float *)malloc(nBytes_band);
  this->reflectance3 = (float *)malloc(nBytes_band);
  this->reflectance4 = (float *)malloc(nBytes_band);
  this->reflectance5 = (float *)malloc(nBytes_band);
  this->reflectance6 = (float *)malloc(nBytes_band);
  this->reflectance7 = (float *)malloc(nBytes_band);

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
  free(this->band1);
  free(this->band2);
  free(this->band3);
  free(this->band4);
  free(this->band5);
  free(this->band6);
  free(this->band7);
  free(this->tal);

  free(this->radiance1);
  free(this->radiance2);
  free(this->radiance3);
  free(this->radiance4);
  free(this->radiance5);
  free(this->radiance6);
  free(this->radiance7);

  free(this->reflectance1);
  free(this->reflectance2);
  free(this->reflectance3);
  free(this->reflectance4);
  free(this->reflectance5);
  free(this->reflectance6);
  free(this->reflectance7);

  free(this->albedo);
  free(this->ndvi);
  free(this->soil_heat);
  free(this->surface_temperature);
  free(this->net_radiation);
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
  free(this->d0);
  free(this->zom);
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
}

void Products::radiance_function(MTL mtl, Sensor sensor)
{
  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  for (int i = 0; i < this->height_band * this->width_band; i++)
  {
    this->radiance1[i] = this->band1[i] * sensor.parameters[1][sensor.GRESCALE] + sensor.parameters[1][sensor.BRESCALE];
    this->radiance2[i] = this->band2[i] * sensor.parameters[2][sensor.GRESCALE] + sensor.parameters[2][sensor.BRESCALE];
    this->radiance3[i] = this->band3[i] * sensor.parameters[3][sensor.GRESCALE] + sensor.parameters[3][sensor.BRESCALE];
    this->radiance4[i] = this->band4[i] * sensor.parameters[4][sensor.GRESCALE] + sensor.parameters[4][sensor.BRESCALE];
    this->radiance5[i] = this->band5[i] * sensor.parameters[5][sensor.GRESCALE] + sensor.parameters[5][sensor.BRESCALE];
    this->radiance6[i] = this->band6[i] * sensor.parameters[6][sensor.GRESCALE] + sensor.parameters[6][sensor.BRESCALE];
    this->radiance7[i] = this->band7[i] * sensor.parameters[7][sensor.GRESCALE] + sensor.parameters[7][sensor.BRESCALE];

    if (radiance1[i] <= 0)
      this->radiance1[i] = NAN;
    if (radiance2[i] <= 0)
      this->radiance2[i] = NAN;
    if (radiance3[i] <= 0)
      this->radiance3[i] = NAN;
    if (radiance4[i] <= 0)
      this->radiance4[i] = NAN;
    if (radiance5[i] <= 0)
      this->radiance5[i] = NAN;
    if (radiance6[i] <= 0)
      this->radiance6[i] = NAN;
    if (radiance7[i] <= 0)
      this->radiance7[i] = NAN;
  }

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
  // std::cout << "SERIAL,RADIANCE," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
}

void Products::reflectance_function(MTL mtl, Sensor sensor)
{
  const float sin_sun = sin(mtl.sun_elevation * PI / 180);

  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  if (mtl.number_sensor == 8)
  {
    for (int i = 0; i < this->height_band * this->width_band; i++)
    {
      this->reflectance1[i] = this->radiance1[i] / sin_sun;
      this->reflectance2[i] = this->radiance2[i] / sin_sun;
      this->reflectance3[i] = this->radiance3[i] / sin_sun;
      this->reflectance4[i] = this->radiance4[i] / sin_sun;
      this->reflectance5[i] = this->radiance5[i] / sin_sun;
      this->reflectance6[i] = this->radiance6[i] / sin_sun;
      this->reflectance7[i] = this->radiance7[i] / sin_sun;
    }
  }
  else
  {
    for (int i = 0; i < this->height_band * this->width_band; i++)
    {
      this->reflectance1[i] = (PI * this->radiance1[i] * mtl.distance_earth_sun * mtl.distance_earth_sun) / (sensor.parameters[1][sensor.ESUN] * sin_sun);
      this->reflectance2[i] = (PI * this->radiance2[i] * mtl.distance_earth_sun * mtl.distance_earth_sun) / (sensor.parameters[2][sensor.ESUN] * sin_sun);
      this->reflectance3[i] = (PI * this->radiance3[i] * mtl.distance_earth_sun * mtl.distance_earth_sun) / (sensor.parameters[3][sensor.ESUN] * sin_sun);
      this->reflectance4[i] = (PI * this->radiance4[i] * mtl.distance_earth_sun * mtl.distance_earth_sun) / (sensor.parameters[4][sensor.ESUN] * sin_sun);
      this->reflectance5[i] = (PI * this->radiance5[i] * mtl.distance_earth_sun * mtl.distance_earth_sun) / (sensor.parameters[5][sensor.ESUN] * sin_sun);
      this->reflectance6[i] = (PI * this->radiance6[i] * mtl.distance_earth_sun * mtl.distance_earth_sun) / (sensor.parameters[6][sensor.ESUN] * sin_sun);
      this->reflectance7[i] = (PI * this->radiance7[i] * mtl.distance_earth_sun * mtl.distance_earth_sun) / (sensor.parameters[7][sensor.ESUN] * sin_sun);
    }
  }

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
  // std::cout << "SERIAL,REFLECTANCE," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
}

void Products::albedo_function(MTL mtl, Sensor sensor)
{
  float *last_band = this->reflectance7;
  float last_param = sensor.parameters[7][sensor.WB];

  if (mtl.number_sensor == 8) {
    last_band = this->reflectance6;
    last_param = sensor.parameters[6][sensor.WB];
  }

  // According to the recommendations of Trezza et al. (2013)
  for (int i = 0; i < this->height_band * this->width_band; i++)
  {
    this->albedo[i] = this->reflectance1[i] * sensor.parameters[1][sensor.WB] +
                      this->reflectance2[i] * sensor.parameters[2][sensor.WB] +
                      this->reflectance3[i] * sensor.parameters[3][sensor.WB] +
                      this->reflectance4[i] * sensor.parameters[4][sensor.WB] +
                      this->reflectance5[i] * sensor.parameters[5][sensor.WB] +
                      last_band[i] * last_param;
  }
}

void Products::ndvi_function()
{
  for (int i = 0; i < this->height_band * this->width_band; i++)
    this->ndvi[i] = (this->reflectance4[i] - this->reflectance3[i]) / (this->reflectance4[i] + this->reflectance3[i]);
};

void Products::pai_function()
{
  for (int i = 0; i < this->height_band * this->width_band; i++)
  {
    double pai_value = 10.1 * (this->reflectance4[i] - sqrt(this->reflectance3[i])) + 3.1;

    if (pai_value < 0)
      pai_value = 0;

    this->pai[i] = pai_value;
  }
};

void Products::lai_function()
{
  for (int i = 0; i < this->height_band * this->width_band; i++)
  {
    float savi= ((1 + 0.05) * (this->reflectance4[i] - this->reflectance3[i])) / (0.05 + (this->reflectance4[i] + this->reflectance3[i]));
    this->savi[i]  = savi;

    if (!isnan(savi) && savi > 0.687)
      this->lai[i] = 6;
    if (!isnan(savi) && savi <= 0.687)
      this->lai[i] = -log((0.69 - savi) / 0.59) / 0.91;
    if (!isnan(savi) && savi < 0.1)
      this->lai[i] = 0;
  }
};

void Products::evi_function()
{
  for (int i = 0; i < this->height_band * this->width_band; i++)
  {
    double evi_value = 2.5 * ((this->reflectance4[i] - this->reflectance3[i]) / (this->reflectance4[i] + (6 * this->reflectance3[i]) - (7.5 * this->reflectance1[i]) + 1));

    if (evi_value < 0)
      evi_value = 0;

    this->evi[i] = evi_value;
  }
};

void Products::enb_emissivity_function()
{
  for (int i = 0; i < this->height_band * this->width_band; i++)
  {
    if (this->lai[i] == 0) 
      this->enb_emissivity[i] = NAN;
    else
      this->enb_emissivity[i] = 0.97 + 0.0033 * this->lai[i];
  
    if ((ndvi[i] < 0) || (lai[i] > 2.99))
      this->enb_emissivity[i] = 0.98;
  }
};

void Products::eo_emissivity_function()
{
  for (int i = 0; i < this->height_band * this->width_band; i++)
  {
    this->eo_emissivity[i] = 0.95 + 0.01 * this->lai[i];

    if (definitelyLessThan(this->ndvi[i], 0) || definitelyGreaterThan(this->lai[i], 2.99))
      this->eo_emissivity[i] = 0.98;
  }
};

void Products::ea_emissivity_function()
{
  for (int i = 0; i < this->height_band * this->width_band; i++)
    this->ea_emissivity[i] = 0.85 * pow((-1 * log(this->tal[i])), 0.09);
};

void Products::surface_temperature_function(MTL mtl)
{
  double k1, k2;
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

  float surface_temperature_value;
  for (int i = 0; i < this->height_band * this->width_band; i++)
  {
    surface_temperature_value = k2 / (log((this->enb_emissivity[i] * k1 / this->radiance6[i]) + 1));

    if (definitelyLessThan(surface_temperature_value, 0))
      surface_temperature_value = 0;

    this->surface_temperature[i] = surface_temperature_value;
  }
};

void Products::short_wave_radiation_function(MTL mtl)
{
  double costheta = sin(mtl.sun_elevation * PI / 180);

  for (int i = 0; i < this->height_band * this->width_band; i++)
    this->short_wave_radiation[i] = (1367 * costheta * this->tal[i]) / (mtl.distance_earth_sun * mtl.distance_earth_sun);
};

void Products::large_wave_radiation_surface_function()
{
  for (int i = 0; i < this->height_band * this->width_band; i++)
  {
    double temperature_pixel = this->surface_temperature[i];
    double surface_temperature_pow_4 = temperature_pixel * temperature_pixel * temperature_pixel * temperature_pixel;
    this->large_wave_radiation_surface[i] = this->eo_emissivity[i] * 5.67 * 1e-8 * surface_temperature_pow_4;
  }
};

void Products::large_wave_radiation_atmosphere_function(double temperature)
{
  double temperature_kelvin = temperature + 273.15;
  double temperature_kelvin_pow_4 = temperature_kelvin * temperature_kelvin * temperature_kelvin * temperature_kelvin;

  for (int i = 0; i < this->height_band * this->width_band; i++)
    this->large_wave_radiation_atmosphere[i] = this->ea_emissivity[i] * 5.67 * 1e-8 * temperature_kelvin_pow_4;
};

void Products::net_radiation_function()
{
  for (int i = 0; i < this->height_band * this->width_band; i++)
  {
    this->net_radiation[i] = this->short_wave_radiation[i] -
                             (this->short_wave_radiation[i] * this->albedo[i]) +
                             this->large_wave_radiation_atmosphere[i] - this->large_wave_radiation_surface[i] -
                             (1 - this->eo_emissivity[i]) * this->large_wave_radiation_atmosphere[i];

    if (definitelyLessThan(this->net_radiation[i], 0))
      this->net_radiation[i] = 0;
  }
};

void Products::soil_heat_flux_function()
{
  for (int i = 0; i < this->height_band * this->width_band; i++)
  {
    if (definitelyLessThan(this->ndvi[i], 0) || definitelyGreaterThan(this->ndvi[i], 0))
    {
      double ndvi_pixel_pow_4 = this->ndvi[i] * this->ndvi[i] * this->ndvi[i] * this->ndvi[i];
      this->soil_heat[i] = (this->surface_temperature[i] - 273.15) * (0.0038 + 0.0074 * this->albedo[i]) *
                           (1 - 0.98 * ndvi_pixel_pow_4) * this->net_radiation[i];
    }
    else
      this->soil_heat[i] = 0.5 * this->net_radiation[i];

    if (definitelyLessThan(this->soil_heat[i], 0))
      this->soil_heat[i] = 0;
  }
};

void Products::d0_fuction()
{
  double CD1 = 20.6;
  double HGHT = 4;

  for (int i = 0; i < this->height_band * this->width_band; i++)
  {
    double pai = this->pai[i];
    double cd1_pai_root = sqrt(CD1 * pai);

    double DISP = HGHT * ((1 - (1 / cd1_pai_root)) + (pow(exp(1.0), -cd1_pai_root) / cd1_pai_root));
    if (pai < 0)
    {
      DISP = 0;
    }

    this->d0[i] = DISP;
  }
};

void Products::kb_function(double ndvi_max, double ndvi_min)
{
  double HGHT = 4;

  double visc = 0.00001461;
  double pr = 0.71;
  double c1 = 0.320;
  double c2 = 0.264;
  double c3 = 15.1;
  double cd = 0.2;
  double ct = 0.01;
  double sf_c = 0.3;
  double sf_d = 2.5;
  double sf_e = 4.0;

  for (int i = 0; i < this->height_band * this->width_band; i++)
  {
    double zom = this->zom[i];
    double u_ast_ini_terra = this->ustar[i];
    double PAI = this->pai[i];

    double Re_star = (u_ast_ini_terra * 0.009) / visc;
    double Ct_star = pow(pr, -0.667) * pow(Re_star, -0.5);
    double beta = c1 - c2 * (exp((cd * -c3 * PAI)));
    double nec_terra = (cd * PAI) / (beta * beta * 2);

    double kb1_fst_part = (cd * VON_KARMAN) / (4 * ct * beta * (1 - exp(nec_terra * -0.5)));
    double kb1_sec_part = (beta * VON_KARMAN * (zom / HGHT)) / Ct_star;
    double kb1s = (pow(Re_star, 0.25) * 2.46) - 2;

    double fc = 1 - pow((this->ndvi[i] - ndvi_max) / (ndvi_min - ndvi_max), 0.4631);
    double fs = 1 - fc;

    double soil_moisture_day_rel = 0.33;

    double SF = sf_c + (1 / (1 + pow(exp(1.0), (sf_d - (sf_e * soil_moisture_day_rel)))));

    this->kb1[i] = ((kb1_fst_part * pow(fc, 2)) +
                    (kb1_sec_part * pow(fc, 2) * pow(fs, 2)) +
                    (pow(fs, 2) * kb1s)) *
                   SF;
  }
};

void Products::zom_fuction(double A_ZOM, double B_ZOM)
{
  double HGHT = 4;
  double CD = 0.01;
  double CR = 0.35;
  double PSICORR = 0.2;

  double gama;
  for (int i = 0; i < this->height_band * this->width_band; i++)
  {
    gama = pow((CD + CR * (this->pai[i] / 2)), -0.5);

    if (gama < 3.3)
      gama = 3.3;

    this->zom[i] = (HGHT - this->d0[i]) * pow(exp(1.0), (-VON_KARMAN * gama) + PSICORR);
  }
};

void Products::ustar_fuction(double u10)
{
  double zu = 10;

  for (int i = 0; i < this->height_band * this->width_band; i++)
  {
    double DISP = this->d0[i];
    double zom = this->zom[i];
    this->ustar[i] = (u10 * VON_KARMAN) / std::log((zu - DISP) / zom);
  }
};

void Products::aerodynamic_resistance_fuction()
{
  double zu = 10.0;

  for (int i = 0; i < this->height_band * this->width_band; i++)
  {
    double DISP = this->d0[i];
    double zom = this->zom[i];
    double zoh_terra = zom / pow(exp(1.0), (this->kb1[i]));

    double temp_kb_1_terra = log(zom / zoh_terra);
    double temp_rah1_terra = (1 / (this->ustar[i] * VON_KARMAN));
    double temp_rah2 = log(((zu - DISP) / zom));
    double temp_rah3_terra = temp_rah1_terra * temp_kb_1_terra;

    this->aerodynamic_resistance[i] = temp_rah1_terra * temp_rah2 + temp_rah3_terra;
  }
};

void Products::sensible_heat_flux_function(double a, double b)
{
  for (int i = 0; i < this->height_band * this->width_band; i++)
  {
    this->sensible_heat_flux[i] = RHO * SPECIFIC_HEAT_AIR * (a + b * (this->surface_temperature[i] - 273.15)) / this->aerodynamic_resistance[i];

    if (!isnan(this->sensible_heat_flux[i]) && definitelyGreaterThan(this->sensible_heat_flux[i], (this->net_radiation[i] - this->soil_heat[i])))
    {
      this->sensible_heat_flux[i] = this->net_radiation[i] - this->soil_heat[i];
    }
  }
};

void Products::latent_heat_flux_function()
{
  for (int i = 0; i < this->height_band * this->width_band; i++)
    this->latent_heat_flux[i] = this->net_radiation[i] - this->soil_heat[i] - this->sensible_heat_flux[i];
};

void Products::net_radiation_24h_function(double Ra24h, double Rs24h)
{
  int FL = 110;

  for (int i = 0; i < this->height_band * this->width_band; i++)
    this->net_radiation_24h[i] = (1 - this->albedo[i]) * Rs24h - FL * Rs24h / Ra24h;
};

void Products::evapotranspiration_fraction_fuction()
{
  for (int i = 0; i < this->height_band * this->width_band; i++)
    this->evapotranspiration_fraction[i] = this->latent_heat_flux[i] / (this->net_radiation[i] - this->soil_heat[i]);
};

void Products::sensible_heat_flux_24h_fuction()
{
  for (int i = 0; i < this->height_band * this->width_band; i++)
    this->sensible_heat_flux_24h[i] = (1 - this->evapotranspiration_fraction[i]) * this->net_radiation_24h[i];
};

void Products::latent_heat_flux_24h_function()
{
  for (int i = 0; i < this->height_band * this->width_band; i++)
    this->latent_heat_flux_24h[i] = this->evapotranspiration_fraction[i] * this->net_radiation_24h[i];
};

void Products::evapotranspiration_24h_function(Station station)
{
  for (int i = 0; i < this->height_band * this->width_band; i++)
    this->evapotranspiration_24h[i] = (this->latent_heat_flux_24h[i] * 86400) / ((2.501 - 0.00236 * (station.v7_max + station.v7_min) / 2) * 1e+6);
};

void Products::evapotranspiration_function()
{
  for (int i = 0; i < this->height_band * this->width_band; i++)
    this->evapotranspiration[i] = this->net_radiation_24h[i] * this->evapotranspiration_fraction[i] * 0.035;
};

string Products::rah_correction_function_serial(double ndvi_min, double ndvi_max, Candidate hot_pixel, Candidate cold_pixel)
{
  system_clock::time_point begin_core, end_core;
  int64_t general_time_core, initial_time_core, final_time_core;

  double hot_pixel_aerodynamic = aerodynamic_resistance[hot_pixel.line * width_band + hot_pixel.col];
  hot_pixel.aerodynamic_resistance.push_back(hot_pixel_aerodynamic);

  double cold_pixel_aerodynamic = aerodynamic_resistance[cold_pixel.line * width_band + cold_pixel.col];
  cold_pixel.aerodynamic_resistance.push_back(cold_pixel_aerodynamic);

  double fc_hot = 1 - pow((ndvi[hot_pixel.line * width_band + hot_pixel.col] - ndvi_max) / (ndvi_min - ndvi_max), 0.4631);
  double fc_cold = 1 - pow((ndvi[cold_pixel.line * width_band + cold_pixel.col] - ndvi_max) / (ndvi_min - ndvi_max), 0.4631);

  for (int i = 0; i < 2; i++)
  {
    this->rah_ini_pq_terra = hot_pixel.aerodynamic_resistance[i];
    this->rah_ini_pf_terra = cold_pixel.aerodynamic_resistance[i];

    double LEc_terra = 0.55 * fc_hot * (hot_pixel.net_radiation - hot_pixel.soil_heat_flux) * 0.78;
    double LEc_terra_pf = 1.75 * fc_cold * (cold_pixel.net_radiation - cold_pixel.soil_heat_flux) * 0.78;

    this->H_pf_terra = cold_pixel.net_radiation - cold_pixel.soil_heat_flux - LEc_terra_pf;
    double dt_pf_terra = H_pf_terra * rah_ini_pf_terra / (RHO * SPECIFIC_HEAT_AIR);

    this->H_pq_terra = hot_pixel.net_radiation - hot_pixel.soil_heat_flux - LEc_terra;
    double dt_pq_terra = H_pq_terra * rah_ini_pq_terra / (RHO * SPECIFIC_HEAT_AIR);

    double b = (dt_pq_terra - dt_pf_terra) / (hot_pixel.temperature - cold_pixel.temperature);
    double a = dt_pf_terra - (b * (cold_pixel.temperature - 273.15));

    // ==== Paralelization core
    begin_core = system_clock::now();
    initial_time_core = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    rah_correction_cycle_STEEP(0, height_band, width_band, a, b, surface_temperature,
                               d0, zom, kb1, sensible_heat_flux, ustar,
                               aerodynamic_resistance);

    end_core = system_clock::now();
    general_time_core = duration_cast<nanoseconds>(end_core - begin_core).count();
    final_time_core = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
    // ==== Paralelization core

    double rah_hot = this->aerodynamic_resistance[hot_pixel.line * width_band + hot_pixel.col];
    hot_pixel.aerodynamic_resistance.push_back(rah_hot);

    double rah_cold = this->aerodynamic_resistance[cold_pixel.line * width_band + cold_pixel.col];
    cold_pixel.aerodynamic_resistance.push_back(rah_cold);
  }

  return "P2 - RAH - PARALLEL - CORE, " + to_string(general_time_core) + ", " + to_string(initial_time_core) + ", " + to_string(final_time_core) + "\n";
}

string Products::rah_correction_function_threads(double ndvi_min, double ndvi_max, Candidate hot_pixel, Candidate cold_pixel)
{
  system_clock::time_point begin_core, end_core;
  int64_t general_time_core, initial_time_core, final_time_core;

  // ==== Threads setup
  int lines_per_thread = height_band / threads_num;
  thread threads[threads_num];
  // ==== Threads setup

  double hot_pixel_aerodynamic = aerodynamic_resistance[hot_pixel.line * width_band + hot_pixel.col];
  hot_pixel.aerodynamic_resistance.push_back(hot_pixel_aerodynamic);

  double cold_pixel_aerodynamic = aerodynamic_resistance[cold_pixel.line * width_band + cold_pixel.col];
  cold_pixel.aerodynamic_resistance.push_back(cold_pixel_aerodynamic);

  double fc_hot = 1 - pow((ndvi[hot_pixel.line * width_band + hot_pixel.col] - ndvi_max) / (ndvi_min - ndvi_max), 0.4631);
  double fc_cold = 1 - pow((ndvi[cold_pixel.line * width_band + cold_pixel.col] - ndvi_max) / (ndvi_min - ndvi_max), 0.4631);

  for (int i = 0; i < 2; i++)
  {
    this->rah_ini_pq_terra = hot_pixel.aerodynamic_resistance[i];
    this->rah_ini_pf_terra = cold_pixel.aerodynamic_resistance[i];

    double LEc_terra = 0.55 * fc_hot * (hot_pixel.net_radiation - hot_pixel.soil_heat_flux) * 0.78;
    double LEc_terra_pf = 1.75 * fc_cold * (cold_pixel.net_radiation - cold_pixel.soil_heat_flux) * 0.78;

    this->H_pf_terra = cold_pixel.net_radiation - cold_pixel.soil_heat_flux - LEc_terra_pf;
    double dt_pf_terra = H_pf_terra * rah_ini_pf_terra / (RHO * SPECIFIC_HEAT_AIR);

    this->H_pq_terra = hot_pixel.net_radiation - hot_pixel.soil_heat_flux - LEc_terra;
    double dt_pq_terra = H_pq_terra * rah_ini_pq_terra / (RHO * SPECIFIC_HEAT_AIR);

    double b = (dt_pq_terra - dt_pf_terra) / (hot_pixel.temperature - cold_pixel.temperature);
    double a = dt_pf_terra - (b * (cold_pixel.temperature - 273.15));

    // ==== Paralelization core
    begin_core = system_clock::now();
    initial_time_core = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    for (int j = 0; j < threads_num; j++)
    {
      int start_line = j * lines_per_thread;
      int end_line = (j == threads_num - 1) ? height_band : (j + 1) * lines_per_thread;
      threads[j] = thread(rah_correction_cycle_STEEP, start_line, end_line, width_band, a, b, surface_temperature,
                          d0, zom, kb1, sensible_heat_flux, ustar,
                          aerodynamic_resistance);
    }

    for (int j = 0; j < threads_num; j++)
      threads[j].join();

    end_core = system_clock::now();
    general_time_core = duration_cast<nanoseconds>(end_core - begin_core).count();
    final_time_core = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
    // ==== Paralelization core

    double rah_hot = this->aerodynamic_resistance[hot_pixel.line * width_band + hot_pixel.col];
    hot_pixel.aerodynamic_resistance.push_back(rah_hot);

    double rah_cold = this->aerodynamic_resistance[cold_pixel.line * width_band + cold_pixel.col];
    cold_pixel.aerodynamic_resistance.push_back(rah_cold);
  }

  return "P2 - RAH - PARALLEL - CORE, " + to_string(general_time_core) + ", " + to_string(initial_time_core) + ", " + to_string(final_time_core) + "\n";
}