#include "kernels.cuh"

__global__ void rad_kernel(float *band_blue_d, float *band_green_d, float *band_red_d, float *band_nir_d, float *band_swir1_d, float *band_termal_d, float *band_swir2_d,
                           float *radiance_blue_d, float *radiance_green_d, float *radiance_red_d, float *radiance_nir_d, float *radiance_swir1_d, float *radiance_termal_d, float *radiance_swir2_d,
                           float *rad_add_d, float *rad_mult_d, int width, int height)
{
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // Map 1D position to 2D grid
  unsigned int row = idx / width;
  unsigned int col = idx % width;

  if (col < width && row < height)
  {
    unsigned int pos = row * width + col;

    radiance_blue_d[pos] = band_blue_d[pos] * rad_mult_d[PARAM_BAND_BLUE_INDEX] + rad_add_d[PARAM_BAND_BLUE_INDEX];
    radiance_green_d[pos] = band_green_d[pos] * rad_mult_d[PARAM_BAND_GREEN_INDEX] + rad_add_d[PARAM_BAND_GREEN_INDEX];
    radiance_red_d[pos] = band_red_d[pos] * rad_mult_d[PARAM_BAND_RED_INDEX] + rad_add_d[PARAM_BAND_RED_INDEX];
    radiance_nir_d[pos] = band_nir_d[pos] * rad_mult_d[PARAM_BAND_NIR_INDEX] + rad_add_d[PARAM_BAND_NIR_INDEX];
    radiance_swir1_d[pos] = band_swir1_d[pos] * rad_mult_d[PARAM_BAND_SWIR1_INDEX] + rad_add_d[PARAM_BAND_SWIR1_INDEX];
    radiance_termal_d[pos] = band_termal_d[pos] * rad_mult_d[PARAM_BAND_TERMAL_INDEX] + rad_add_d[PARAM_BAND_TERMAL_INDEX];
    radiance_swir2_d[pos] = band_swir2_d[pos] * rad_mult_d[PARAM_BAND_SWIR2_INDEX] + rad_add_d[PARAM_BAND_SWIR2_INDEX];

    if (radiance_blue_d[pos] <= 0)
      radiance_blue_d[pos] = NAN;
    if (radiance_green_d[pos] <= 0)
      radiance_green_d[pos] = NAN;
    if (radiance_red_d[pos] <= 0)
      radiance_red_d[pos] = NAN;
    if (radiance_nir_d[pos] <= 0)
      radiance_nir_d[pos] = NAN;
    if (radiance_swir1_d[pos] <= 0)
      radiance_swir1_d[pos] = NAN;
    if (radiance_termal_d[pos] <= 0)
      radiance_termal_d[pos] = NAN;
    if (radiance_swir2_d[pos] <= 0)
      radiance_swir2_d[pos] = NAN;
  }
}

__global__ void ref_kernel(float *band_blue_d, float *band_green_d, float *band_red_d, float *band_nir_d, float *band_swir1_d, float *band_termal_d, float *band_swir2_d,
                           float *reflectance_blue_d, float *reflectance_green_d, float *reflectance_red_d, float *reflectance_nir_d, float *reflectance_swir1_d, float *reflectance_termal_d, float *reflectance_swir2_d,
                           float *ref_add_d, float *ref_mult_d, float sin_sun, int width, int height)
{
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // Map 1D position to 2D grid
  unsigned int row = idx / width;
  unsigned int col = idx % width;

  if (col < width && row < height)
  {
    unsigned int pos = row * width + col;

    reflectance_blue_d[pos] = (band_blue_d[pos] * ref_mult_d[PARAM_BAND_BLUE_INDEX] + ref_add_d[PARAM_BAND_BLUE_INDEX]) / sin_sun;
    reflectance_green_d[pos] = (band_green_d[pos] * ref_mult_d[PARAM_BAND_GREEN_INDEX] + ref_add_d[PARAM_BAND_GREEN_INDEX]) / sin_sun;
    reflectance_red_d[pos] = (band_red_d[pos] * ref_mult_d[PARAM_BAND_RED_INDEX] + ref_add_d[PARAM_BAND_RED_INDEX]) / sin_sun;
    reflectance_nir_d[pos] = (band_nir_d[pos] * ref_mult_d[PARAM_BAND_NIR_INDEX] + ref_add_d[PARAM_BAND_NIR_INDEX]) / sin_sun;
    reflectance_swir1_d[pos] = (band_swir1_d[pos] * ref_mult_d[PARAM_BAND_SWIR1_INDEX] + ref_add_d[PARAM_BAND_SWIR1_INDEX]) / sin_sun;
    reflectance_termal_d[pos] = (band_termal_d[pos] * ref_mult_d[PARAM_BAND_TERMAL_INDEX] + ref_add_d[PARAM_BAND_TERMAL_INDEX]) / sin_sun;
    reflectance_swir2_d[pos] = (band_swir2_d[pos] * ref_mult_d[PARAM_BAND_SWIR2_INDEX] + ref_add_d[PARAM_BAND_SWIR2_INDEX]) / sin_sun;

    if (reflectance_blue_d[pos] <= 0)
      reflectance_blue_d[pos] = NAN;
    if (reflectance_green_d[pos] <= 0)
      reflectance_green_d[pos] = NAN;
    if (reflectance_red_d[pos] <= 0)
      reflectance_red_d[pos] = NAN;
    if (reflectance_nir_d[pos] <= 0)
      reflectance_nir_d[pos] = NAN;
    if (reflectance_swir1_d[pos] <= 0)
      reflectance_swir1_d[pos] = NAN;
    if (reflectance_termal_d[pos] <= 0)
      reflectance_termal_d[pos] = NAN;
    if (reflectance_swir2_d[pos] <= 0)
      reflectance_swir2_d[pos] = NAN;
  }
}

__global__ void albedo_kernel(float *reflectance_blue_d, float *reflectance_green_d, float *reflectance_red_d, float *reflectance_nir_d, float *reflectance_swir1_d, float *reflectance_swir2_d,
                              float *tal_d, float *albedo_d, float *ref_w_coeff_d, int width, int height)
{
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // Map 1D position to 2D grid
  unsigned int row = idx / width;
  unsigned int col = idx % width;

  if (col < width && row < height)
  {
    unsigned int pos = row * width + col;

    float alb_toa = reflectance_blue_d[pos] * ref_w_coeff_d[PARAM_BAND_BLUE_INDEX] +
                    reflectance_green_d[pos] * ref_w_coeff_d[PARAM_BAND_GREEN_INDEX] +
                    reflectance_red_d[pos] * ref_w_coeff_d[PARAM_BAND_RED_INDEX] +
                    reflectance_nir_d[pos] * ref_w_coeff_d[PARAM_BAND_NIR_INDEX] +
                    reflectance_swir1_d[pos] * ref_w_coeff_d[PARAM_BAND_SWIR1_INDEX] +
                    reflectance_swir2_d[pos] * ref_w_coeff_d[PARAM_BAND_SWIR2_INDEX];

    albedo_d[pos] = (alb_toa - 0.03) / (tal_d[pos] * tal_d[pos]);

    if (albedo_d[pos] <= 0)
      albedo_d[pos] = NAN;
  }
}

__global__ void ndvi_kernel(float *band_nir_d, float *band_red_d, float *ndvi_d, int width, int height)
{
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // Map 1D position to 2D grid
  unsigned int row = idx / width;
  unsigned int col = idx % width;

  if (col < width && row < height)
  {
    unsigned int pos = row * width + col;

    ndvi_d[pos] = (band_nir_d[pos] - band_red_d[pos]) / (band_nir_d[pos] + band_red_d[pos]);

    if (ndvi_d[pos] <= -1 || ndvi_d[pos] >= 1)
      ndvi_d[pos] = NAN;
  }
}

__global__ void pai_kernel(float *band_nir_d, float *band_red_d, float *pai_d, int width, int height)
{
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // Map 1D position to 2D grid
  unsigned int row = idx / width;
  unsigned int col = idx % width;

  if (col < width && row < height)
  {
    unsigned int pos = row * width + col;

    pai_d[pos] = 10.1 * (band_nir_d[pos] - sqrt(band_red_d[pos])) + 3.1;

    if (pai_d[pos] <= 0)
      pai_d[pos] = NAN;
  }
}

__global__ void lai_kernel(float *reflectance_nir_d, float *reflectance_red_d, float *lai_d, int width_band, int height_band)
{
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // Map 1D position to 2D grid
  unsigned int row = idx / width_band;
  unsigned int col = idx % width_band;

  if (col < width_band && row < height_band)
  {
    unsigned int pos = row * width_band + col;

    float savi = ((1 + 0.5) * (reflectance_nir_d[pos] - reflectance_red_d[pos])) / (0.5 + (reflectance_nir_d[pos] + reflectance_red_d[pos]));

    if (!isnan(savi) && savi > 0.687)
      lai_d[pos] = 6;
    if (!isnan(savi) && savi <= 0.687)
      lai_d[pos] = -log((0.69 - savi) / 0.59) / 0.91;
    if (!isnan(savi) && savi < 0.1)
      lai_d[pos] = 0;
  }
}

__global__ void evi_kernel(float *reflectance_nir_d, float *reflectance_red_d, float *reflectance_blue_d, float *evi_d, int width_band, int height_band)
{
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // Map 1D position to 2D grid
  unsigned int row = idx / width_band;
  unsigned int col = idx % width_band;

  if (col < width_band && row < height_band)
  {
    unsigned int pos = row * width_band + col;

    evi_d[pos] = 2.5 * ((reflectance_nir_d[pos] - reflectance_red_d[pos]) / (reflectance_nir_d[pos] + (6 * reflectance_red_d[pos]) - (7.5 * reflectance_blue_d[pos]) + 1));

    if (evi_d[pos] < 0)
      evi_d[pos] = NAN;
  }
}

__global__ void enb_kernel(float *lai_d, float *enb_d, int width_band, int height_band)
{
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // Map 1D position to 2D grid
  unsigned int row = idx / width_band;
  unsigned int col = idx % width_band;

  if (col < width_band && row < height_band)
  {
    unsigned int pos = row * width_band + col;

    if (lai_d[pos] == 0)
      enb_d[pos] = NAN;
    else
      enb_d[pos] = 0.97 + 0.0033 * lai_d[pos];

    if (enb_d[pos] < 0 || lai_d[pos] > 2.99)
      enb_d[pos] = 0.98;
  }
}

__global__ void eo_kernel(float *lai_d, float *ndvi_d, float *eo_d, int width_band, int height_band)
{
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // Map 1D position to 2D grid
  unsigned int row = idx / width_band;
  unsigned int col = idx % width_band;

  if (col < width_band && row < height_band)
  {
    unsigned int pos = row * width_band + col;

    if (lai_d[pos] == 0)
      eo_d[pos] = NAN;
    else
      eo_d[pos] = 0.95 + 0.01 * lai_d[pos];

    if (ndvi_d[pos] < 0 || lai_d[pos] > 2.99)
      eo_d[pos] = 0.98;
  }
}

__global__ void ea_kernel(float *tal_d, float *ea_d, int width_band, int height_band)
{
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // Map 1D position to 2D grid
  unsigned int row = idx / width_band;
  unsigned int col = idx % width_band;

  if (col < width_band && row < height_band)
  {
    unsigned int pos = row * width_band + col;
    ea_d[pos] = 0.85 * pow((-1 * log(tal_d[pos])), 0.09);
  }
}

__global__ void surface_temperature_kernel(float *enb_d, float *radiance_termal_d, float *surface_temperature_d, float k1, float k2, int width_band, int height_band)
{
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // Map 1D position to 2D grid
  unsigned int row = idx / width_band;
  unsigned int col = idx % width_band;

  if (col < width_band && row < height_band)
  {
    unsigned int pos = row * width_band + col;
    surface_temperature_d[pos] = k2 / (log((enb_d[pos] * k1 / radiance_termal_d[pos]) + 1));

    if (surface_temperature_d[pos] < 0)
      surface_temperature_d[pos] = 0;
  }
}

__global__ void short_wave_radiation_kernel(float *tal_d, float *short_wave_radiation_d, float sun_elevation, float distance_earth_sun, float pi, int width_band, int height_band)
{
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // Map 1D position to 2D grid
  unsigned int row = idx / width_band;
  unsigned int col = idx % width_band;

  if (col < width_band && row < height_band)
  {
    unsigned int pos = row * width_band + col;
    short_wave_radiation_d[pos] = (1367 * sin(sun_elevation * pi / 180) * tal_d[pos]) / (distance_earth_sun * distance_earth_sun);
  }
}

__global__ void large_wave_radiation_surface_kernel(float *surface_temperature_d, float *eo_d, float *large_wave_radiation_surface_d, int width_band, int height_band)
{
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // Map 1D position to 2D grid
  unsigned int row = idx / width_band;
  unsigned int col = idx % width_band;

  if (col < width_band && row < height_band)
  {
    unsigned int pos = row * width_band + col;
    float temperature_pixel = surface_temperature_d[pos];
    float surface_temperature_pow_4 = temperature_pixel * temperature_pixel * temperature_pixel * temperature_pixel;
    large_wave_radiation_surface_d[pos] = eo_d[pos] * 5.67 * 1e-8 * surface_temperature_pow_4;
  }
}

__global__ void large_wave_radiation_atmosphere_kernel(float *ea_d, float *large_wave_radiation_atmosphere_d, float temperature, int width_band, int height_band)
{
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // Map 1D position to 2D grid
  unsigned int row = idx / width_band;
  unsigned int col = idx % width_band;

  if (col < width_band && row < height_band)
  {
    unsigned int pos = row * width_band + col;
    float temperature_kelvin = temperature + 273.15;
    float temperature_kelvin_pow_4 = temperature_kelvin * temperature_kelvin * temperature_kelvin * temperature_kelvin;

    large_wave_radiation_atmosphere_d[pos] = ea_d[pos] * 5.67 * 1e-8 * temperature_kelvin_pow_4;
  }
}

__global__ void net_radiation_kernel(float *short_wave_radiation_d, float *albedo_d, float *large_wave_radiation_atmosphere_d, float *large_wave_radiation_surface_d, float *eo_d, float *net_radiation_d, int width_band, int height_band)
{
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // Map 1D position to 2D grid
  unsigned int row = idx / width_band;
  unsigned int col = idx % width_band;

  if (col < width_band && row < height_band)
  {
    unsigned int pos = row * width_band + col;
    net_radiation_d[pos] = short_wave_radiation_d[pos] - (short_wave_radiation_d[pos] * albedo_d[pos]) + large_wave_radiation_atmosphere_d[pos] - large_wave_radiation_surface_d[pos] - (1 - eo_d[pos]) * large_wave_radiation_atmosphere_d[pos];

    if (net_radiation_d[pos] < 0)
      net_radiation_d[pos] = 0;
  }
}

__global__ void soil_heat_kernel(float *ndvi_d, float *albedo_d, float *surface_temperature_d, float *net_radiation_d, float *soil_heat_d, int width_band, int height_band)
{
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // Map 1D position to 2D grid
  unsigned int row = idx / width_band;
  unsigned int col = idx % width_band;

  if (col < width_band && row < height_band)
  {
    unsigned int pos = row * width_band + col;

    if (ndvi_d[pos] < 0 || ndvi_d[pos] > 0)
    {
      float ndvi_pixel_pow_4 = ndvi_d[pos] * ndvi_d[pos] * ndvi_d[pos] * ndvi_d[pos];
      soil_heat_d[pos] = (surface_temperature_d[pos] - 273.15) * (0.0038 + 0.0074 * albedo_d[pos]) * (1 - 0.98 * ndvi_pixel_pow_4) * net_radiation_d[pos];
    }
    else
    {
      soil_heat_d[pos] = 0.5 * net_radiation_d[pos];
    }

    if (soil_heat_d[pos] < 0)
      soil_heat_d[pos] = 0;
  }
}

__global__ void d0_kernel(float *pai_d, float *d0_d, float CD1, float HGHT, int width_band, int height_band)
{
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // Map 1D position to 2D grid
  unsigned int row = idx / width_band;
  unsigned int col = idx % width_band;

  if (col < width_band && row < height_band)
  {
    unsigned int pos = row * width_band + col;
    float cd1_pai_root = sqrt(CD1 * pai_d[pos]);

    float DISP = HGHT * ((1 - (1 / cd1_pai_root)) + (pow(exp(1.0), -cd1_pai_root) / cd1_pai_root));
    if (pai_d[pos] < 0)
    {
      DISP = 0;
    }

    d0_d[pos] = DISP;
  }
}

__global__ void ustar_kernel(float *zom_d, float *d0_d, float *ustarR_d, float u10, int width_band, int height_band)
{
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // Map 1D position to 2D grid
  unsigned int row = idx / width_band;
  unsigned int col = idx % width_band;

  float zu = 10;
  if (col < width_band && row < height_band)
  {
    unsigned int pos = row * width_band + col;
    ustarR_d[pos] = (u10 * VON_KARMAN) / log((zu - d0_d[pos]) / zom_d[pos]);
  }
}

__global__ void zom_kernel(float *d0_d, float *pai_d, float *zom_d, float A_ZOM, float B_ZOM, int width_band, int height_band)
{
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // Map 1D position to 2D grid
  unsigned int row = idx / width_band;
  unsigned int col = idx % width_band;

  float HGHT = 4;
  float CD = 0.01;
  float CR = 0.35;
  float PSICORR = 0.2;

  if (col < width_band && row < height_band)
  {
    unsigned int pos = row * width_band + col;

    float gama = pow((CD + CR * (pai_d[pos] / 2)), -0.5);

    if (gama < 3.3)
      gama = 3.3;

    zom_d[pos] = (HGHT - d0_d[pos]) * pow(exp(1.0), (-VON_KARMAN * gama) + PSICORR);
  }
}

__global__ void kb_kernel(float *zom_d, float *ustarR_d, float *pai_d, float *kb1_d, float *ndvi_d, int width_band, int height_band, float ndvi_max, float ndvi_min)
{
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // Map 1D position to 2D grid
  unsigned int row = idx / width_band;
  unsigned int col = idx % width_band;

  float HGHT = 4;

  float VON_KARMAN = 0.41;
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
  float soil_moisture_day_rel = 0.33;

  if (col < width_band && row < height_band)
  {
    unsigned int pos = row * width_band + col;

    float Re_star = (ustarR_d[pos] * 0.009) / visc;
    float Ct_star = pow(pr, -0.667) * pow(Re_star, -0.5);
    float beta = c1 - c2 * (exp((cd * -c3 * pai_d[pos])));
    float nec_terra = (cd * pai_d[pos]) / (beta * beta * 2);

    float kb1_fst_part = (cd * VON_KARMAN) / (4 * ct * beta * (1 - exp(nec_terra * -0.5)));
    float kb1_sec_part = (beta * VON_KARMAN * (zom_d[pos] / HGHT)) / Ct_star;
    float kb1s = (pow(Re_star, 0.25) * 2.46) - 2;

    float fc = 1 - pow((ndvi_d[pos] - ndvi_max) / (ndvi_min - ndvi_max), 0.4631);
    float fs = 1 - fc;

    float SF = sf_c + (1 / (1 + pow(exp(1.0), (sf_d - (sf_e * soil_moisture_day_rel)))));

    kb1_d[pos] = ((kb1_fst_part * pow(fc, 2)) + (kb1_sec_part * pow(fc, 2) * pow(fs, 2)) + (pow(fs, 2) * kb1s)) * SF;
  }
}

__global__ void aerodynamic_resistance_kernel(float *zom_d, float *d0_d, float *ustarR_d, float *kb1_d, float *aerodynamic_resistance_d, int width_band, int height_band)
{
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // Map 1D position to 2D grid
  unsigned int row = idx / width_band;
  unsigned int col = idx % width_band;

  float zu = 10.0;

  if (col < width_band && row < height_band)
  {
    unsigned int pos = row * width_band + col;

    float DISP = d0_d[pos];
    float zom = zom_d[pos];
    float zoh_terra = zom / pow(exp(1.0), (kb1_d[pos]));

    float temp_kb_1_terra = log(zom / zoh_terra);
    float temp_rah1_terra = (1 / (ustarR_d[pos] * VON_KARMAN));
    float temp_rah2 = log(((zu - DISP) / zom));
    float temp_rah3_terra = temp_rah1_terra * temp_kb_1_terra;

    aerodynamic_resistance_d[pos] = temp_rah1_terra * temp_rah2 + temp_rah3_terra;
  }
}

__global__ void sensible_heat_flux_kernel(float *surface_temperature_d, float *rahR_d, float *net_radiation_d, float *soil_heat_d, float *sensible_heat_flux_d, float a, float b, int width_band, int height_band)
{
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // Map 1D position to 2D grid
  unsigned int row = idx / width_band;
  unsigned int col = idx % width_band;

  if (col < width_band && row < height_band)
  {
    unsigned int pos = row * width_band + col;

    sensible_heat_flux_d[pos] = RHO * SPECIFIC_HEAT_AIR * (a + b * (surface_temperature_d[pos] - 273.15)) / rahR_d[pos];

    if (!isnan(sensible_heat_flux_d[pos]) && sensible_heat_flux_d[pos] > (net_radiation_d[pos] - soil_heat_d[pos]))
    {
      sensible_heat_flux_d[pos] = net_radiation_d[pos] - soil_heat_d[pos];
    }
  }
}

__global__ void latent_heat_flux_kernel(float *net_radiation_d, float *soil_heat_d, float *sensible_heat_flux_d, float *latent_heat_flux_d, int width_band, int height_band)
{
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // Map 1D position to 2D grid
  unsigned int row = idx / width_band;
  unsigned int col = idx % width_band;

  if (col < width_band && row < height_band)
  {
    unsigned int pos = row * width_band + col;
    latent_heat_flux_d[pos] = net_radiation_d[pos] - soil_heat_d[pos] - sensible_heat_flux_d[pos];
  }
}

__global__ void net_radiation_24h_kernel(float *albedo_d, float Rs24h, float Ra24h, float *net_radiation_24h_d, int width_band, int height_band)
{
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // Map 1D position to 2D grid
  unsigned int row = idx / width_band;
  unsigned int col = idx % width_band;

  int FL = 110;

  if (col < width_band && row < height_band)
  {
    unsigned int pos = row * width_band + col;
    net_radiation_24h_d[pos] = (1 - albedo_d[pos]) * Rs24h - FL * Rs24h / Ra24h;
  }
}

__global__ void evapotranspiration_fraction_kernel(float *net_radiation_d, float *soil_heat_d, float *latent_heat_flux_d, float *evapotranspiration_fraction_d, int width_band, int height_band)
{
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // Map 1D position to 2D grid
  unsigned int row = idx / width_band;
  unsigned int col = idx % width_band;

  if (col < width_band && row < height_band)
  {
    unsigned int pos = row * width_band + col;
    evapotranspiration_fraction_d[pos] = latent_heat_flux_d[pos] / (net_radiation_d[pos] - soil_heat_d[pos]);
  }
}

__global__ void sensible_heat_flux_24h_kernel(float *net_radiation_24h_d, float *evapotranspiration_fraction_d, float *sensible_heat_flux_24h_d, int width_band, int height_band)
{
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // Map 1D position to 2D grid
  unsigned int row = idx / width_band;
  unsigned int col = idx % width_band;

  if (col < width_band && row < height_band)
  {
    unsigned int pos = row * width_band + col;
    sensible_heat_flux_24h_d[pos] = (1 - evapotranspiration_fraction_d[pos]) * net_radiation_24h_d[pos];
  }
}

__global__ void latent_heat_flux_24h_kernel(float *net_radiation_24h_d, float *evapotranspiration_fraction_d, float *latent_heat_flux_24h_d, int width_band, int height_band)
{
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // Map 1D position to 2D grid
  unsigned int row = idx / width_band;
  unsigned int col = idx % width_band;

  if (col < width_band && row < height_band)
  {
    unsigned int pos = row * width_band + col;
    latent_heat_flux_24h_d[pos] = evapotranspiration_fraction_d[pos] * net_radiation_24h_d[pos];
  }
}

__global__ void evapotranspiration_24h_kernel(float *latent_heat_flux_24h_d, float *evapotranspiration_24h_d, float v7_max, float v7_min, int width_band, int height_band)
{
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // Map 1D position to 2D grid
  unsigned int row = idx / width_band;
  unsigned int col = idx % width_band;

  if (col < width_band && row < height_band)
  {
    unsigned int pos = row * width_band + col;
    evapotranspiration_24h_d[pos] = (latent_heat_flux_24h_d[pos] * 86400) / ((2.501 - 0.00236 * (v7_max + v7_min) / 2) * 1e+6);
  }
}

__global__ void evapotranspiration_kernel(float *net_radiation_24h_d, float *evapotranspiration_fraction_d, float *evapotranspiration_d, int width_band, int height_band)
{
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // Map 1D position to 2D grid
  unsigned int row = idx / width_band;
  unsigned int col = idx % width_band;

  if (col < width_band && row < height_band)
  {
    unsigned int pos = row * width_band + col;
    evapotranspiration_d[pos] = net_radiation_24h_d[pos] * evapotranspiration_fraction_d[pos] * 0.035;
  }
}

__global__ void rah_correction_cycle_STEEP(float *surface_temperature_pointer, float *d0_pointer, float *kb1_pointer, float *zom_pointer, float *ustarR_pointer,
                                           float *ustarW_pointer, float *rahR_pointer, float *rahW_pointer, float *H_pointer, float a, float b, int height,
                                           int width)
{
  // Identify 1D position
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // Map 1D position to 2D grid
  unsigned int row = idx / width;
  unsigned int col = idx % width;

  if (col < width && row < height)
  {
    unsigned int pos = row * width + col;

    float DISP = d0_pointer[pos];
    float dT_ini_terra = a + b * (surface_temperature_pointer[pos] - 273.15);

    float sensibleHeatFlux = RHO * SPECIFIC_HEAT_AIR * (dT_ini_terra) / rahR_pointer[pos];
    float L = -1 * ((RHO * SPECIFIC_HEAT_AIR * pow(ustarR_pointer[pos], 3) * surface_temperature_pointer[pos]) / (VON_KARMAN * GRAVITY * sensibleHeatFlux));

    float y2 = pow((1 - (16 * (10 - DISP)) / L), 0.25);
    float x200 = pow((1 - (16 * (10 - DISP)) / L), 0.25);

    float psi2, psi200;
    if (!isnan(L) && L > 0)
    {
      psi2 = -5 * ((10 - DISP) / L);
      psi200 = -5 * ((10 - DISP) / L);
    }
    else
    {
      psi2 = 2 * log((1 + y2 * y2) / 2);
      psi200 = 2 * log((1 + x200) / 2) + log((1 + x200 * x200) / 2) - 2 * atan(x200) + 0.5 * M_PI;
    }

    float ust = (VON_KARMAN * ustarR_pointer[pos]) / (log((10 - DISP) / zom_pointer[pos]) - psi200);

    float zoh_terra = zom_pointer[pos] / pow(exp(1.0), (kb1_pointer[pos]));
    float temp_rah1_corr_terra = (ust * VON_KARMAN);
    float temp_rah2_corr_terra = log((10 - DISP) / zom_pointer[pos]) - psi2;
    float temp_rah3_corr_terra = temp_rah1_corr_terra * log(zom_pointer[pos] / zoh_terra);
    float rah = (temp_rah1_corr_terra * temp_rah2_corr_terra) + temp_rah3_corr_terra;

    ustarW_pointer[pos] = ust;
    rahW_pointer[pos] = rah;
    H_pointer[pos] = sensibleHeatFlux;
  }
}
