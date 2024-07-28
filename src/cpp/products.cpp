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
    this->radiance_blue[i] = this->band_blue[i] * sensor.parameters[1][sensor.GRESCALE] + sensor.parameters[1][sensor.BRESCALE];
    this->radiance_green[i] = this->band_green[i] * sensor.parameters[2][sensor.GRESCALE] + sensor.parameters[2][sensor.BRESCALE];
    this->radiance_red[i] = this->band_red[i] * sensor.parameters[3][sensor.GRESCALE] + sensor.parameters[3][sensor.BRESCALE];
    this->radiance_nir[i] = this->band_nir[i] * sensor.parameters[4][sensor.GRESCALE] + sensor.parameters[4][sensor.BRESCALE];
    this->radiance_swir1[i] = this->band_swir1[i] * sensor.parameters[5][sensor.GRESCALE] + sensor.parameters[5][sensor.BRESCALE];
    this->radiance_termal[i] = this->band_termal[i] * sensor.parameters[6][sensor.GRESCALE] + sensor.parameters[6][sensor.BRESCALE];
    this->radiance_swir2[i] = this->band_swir2[i] * sensor.parameters[7][sensor.GRESCALE] + sensor.parameters[7][sensor.BRESCALE];

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
      this->reflectance_blue[i] = this->radiance_blue[i] / sin_sun;
      this->reflectance_green[i] = this->radiance_green[i] / sin_sun;
      this->reflectance_red[i] = this->radiance_red[i] / sin_sun;
      this->reflectance_nir[i] = this->radiance_nir[i] / sin_sun;
      this->reflectance_swir1[i] = this->radiance_swir1[i] / sin_sun;
      this->reflectance_termal[i] = this->radiance_termal[i] / sin_sun;
      this->reflectance_swir2[i] = this->radiance_swir2[i] / sin_sun;
    }
  }
  else
  {
    for (int i = 0; i < this->height_band * this->width_band; i++)
    {
      this->reflectance_blue[i] = (PI * this->radiance_blue[i]) / (sensor.parameters[1][sensor.ESUN] * sin_sun);
      this->reflectance_green[i] = (PI * this->radiance_green[i]) / (sensor.parameters[2][sensor.ESUN] * sin_sun);
      this->reflectance_red[i] = (PI * this->radiance_red[i]) / (sensor.parameters[3][sensor.ESUN] * sin_sun);
      this->reflectance_nir[i] = (PI * this->radiance_nir[i]) / (sensor.parameters[4][sensor.ESUN] * sin_sun);
      this->reflectance_swir1[i] = (PI * this->radiance_swir1[i]) / (sensor.parameters[5][sensor.ESUN] * sin_sun);
      this->reflectance_termal[i] = (PI * this->radiance_termal[i]) / (sensor.parameters[6][sensor.ESUN] * sin_sun);
      this->reflectance_swir2[i] = (PI * this->radiance_swir2[i]) / (sensor.parameters[7][sensor.ESUN] * sin_sun);
    }
  }

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
  // std::cout << "SERIAL,REFLECTANCE," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
}

void Products::albedo_function(MTL mtl, Sensor sensor)
{
  if (mtl.number_sensor == 8)
  {
    for (int i = 0; i < this->height_band * this->width_band; i++)
    {
      this->albedo[i] = this->reflectance_blue[i] * sensor.parameters[1][sensor.WB] +
                        this->reflectance_green[i] * sensor.parameters[2][sensor.WB] +
                        this->reflectance_red[i] * sensor.parameters[3][sensor.WB] +
                        this->reflectance_nir[i] * sensor.parameters[4][sensor.WB] +
                        this->reflectance_swir1[i] * sensor.parameters[5][sensor.WB] +
                        this->reflectance_termal[i] * sensor.parameters[6][sensor.WB];
    }
  }
  else
  {
    for (int i = 0; i < this->height_band * this->width_band; i++)
    {
      float alb = this->reflectance_blue[i] * sensor.parameters[1][sensor.WB] +
                  this->reflectance_green[i] * sensor.parameters[2][sensor.WB] +
                  this->reflectance_red[i] * sensor.parameters[3][sensor.WB] +
                  this->reflectance_nir[i] * sensor.parameters[4][sensor.WB] +
                  this->reflectance_swir1[i] * sensor.parameters[5][sensor.WB] +
                  this->reflectance_swir2[i] * sensor.parameters[7][sensor.WB];
      this->albedo[i] = (alb - 0.03) / (this->tal[i] * this->tal[i]);
    }
  }
}

void Products::ndvi_function()
{
  for (int i = 0; i < this->height_band * this->width_band; i++)
    this->ndvi[i] = (this->reflectance_nir[i] - this->reflectance_red[i]) / (this->reflectance_nir[i] + this->reflectance_red[i]);
};

void Products::pai_function()
{
  for (int i = 0; i < this->height_band * this->width_band; i++)
  {
    double pai_value = 10.1 * (this->reflectance_nir[i] - sqrt(this->reflectance_red[i])) + 3.1;

    if (pai_value < 0)
      pai_value = 0;

    this->pai[i] = pai_value;
  }
};

void Products::lai_function()
{
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
  }
};

void Products::evi_function()
{
  for (int i = 0; i < this->height_band * this->width_band; i++)
  {
    double evi_value = 2.5 * ((this->reflectance_nir[i] - this->reflectance_red[i]) / (this->reflectance_nir[i] + (6 * this->reflectance_red[i]) - (7.5 * this->reflectance_blue[i]) + 1));

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
    surface_temperature_value = k2 / (log((this->enb_emissivity[i] * k1 / this->radiance_termal[i]) + 1));

    if (definitelyLessThan(surface_temperature_value, 0))
      surface_temperature_value = 0;

    this->surface_temperature[i] = surface_temperature_value;
  }
};

void Products::short_wave_radiation_function(MTL mtl)
{
  float costheta = sin(mtl.sun_elevation * PI / 180);

  for (int i = 0; i < this->height_band * this->width_band; i++)
    this->short_wave_radiation[i] = (1367 * costheta * this->tal[i]) / (mtl.distance_earth_sun * mtl.distance_earth_sun);
};

void Products::large_wave_radiation_surface_function()
{
  for (int i = 0; i < this->height_band * this->width_band; i++)
  {
    float temperature_pixel = this->surface_temperature[i];
    float surface_temperature_pow_4 = temperature_pixel * temperature_pixel * temperature_pixel * temperature_pixel;
    this->large_wave_radiation_surface[i] = this->eo_emissivity[i] * 5.67 * 1e-8 * surface_temperature_pow_4;
  }
};

void Products::large_wave_radiation_atmosphere_function(double temperature)
{
  float temperature_kelvin = temperature + 273.15;
  float temperature_kelvin_pow_4 = temperature_kelvin * temperature_kelvin * temperature_kelvin * temperature_kelvin;

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
      float ndvi_pixel_pow_4 = this->ndvi[i] * this->ndvi[i] * this->ndvi[i] * this->ndvi[i];
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
  float CD1 = 20.6;
  float HGHT = 4;

  for (int i = 0; i < this->height_band * this->width_band; i++)
  {
    float pai = this->pai[i];
    float cd1_pai_root = sqrt(CD1 * pai);

    float DISP = HGHT * ((1 - (1 / cd1_pai_root)) + (pow(exp(1.0), -cd1_pai_root) / cd1_pai_root));
    if (pai < 0)
    {
      DISP = 0;
    }

    this->d0[i] = DISP;
  }
};

void Products::kb_function(double ndvi_max, double ndvi_min)
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
};

void Products::zom_fuction(double A_ZOM, double B_ZOM)
{
  float HGHT = 4;
  float CD = 0.01;
  float CR = 0.35;
  float PSICORR = 0.2;

  float gama;
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
  float zu = 10;

  for (int i = 0; i < this->height_band * this->width_band; i++)
  {
    float DISP = this->d0[i];
    float zom = this->zom[i];
    this->ustar[i] = (u10 * VON_KARMAN) / log((zu - DISP) / zom);
  }
};

void Products::aerodynamic_resistance_fuction()
{
  float zu = 10.0;

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