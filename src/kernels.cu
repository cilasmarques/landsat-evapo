#include "kernels.cuh"

__shared__ float rah_ini_pq_terra;
__shared__ float rah_ini_pf_terra;
__shared__ float H_pq_terra;
__shared__ float H_pf_terra;

__global__ void rad_kernel(float *band_d, float *radiance_d, float *rad_add_d, float *rad_mult_d, int band_idx, int width, int height)
{
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // Map 1D position to 2D grid
  unsigned int row = idx / width;
  unsigned int col = idx % width;

  if (idx < width * height)
  {
    unsigned int pos = row * width + col;

    radiance_d[pos] = band_d[pos] * rad_mult_d[band_idx] + rad_add_d[band_idx];

    if (radiance_d[pos] <= 0)
      radiance_d[pos] = NAN;
  }
}

__global__ void ref_kernel(float *band_d, float *reflectance_d, float *ref_add_d, float *ref_mult_d, float sin_sun, int band_idx, int width, int height)
{
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // Map 1D position to 2D grid
  unsigned int row = idx / width;
  unsigned int col = idx % width;

  if (idx < width * height)
  {
    unsigned int pos = row * width + col;

    reflectance_d[pos] = (band_d[pos] * ref_mult_d[band_idx] + ref_add_d[band_idx]) / sin_sun;

    if (reflectance_d[pos] <= 0)
      reflectance_d[pos] = NAN;
  }
}

__global__ void albedo_kernel(float *reflectance_blue_d, float *reflectance_green_d, float *reflectance_red_d, float *reflectance_nir_d, float *reflectance_swir1_d, float *reflectance_swir2_d, float *tal_d, float *albedo_d, float *ref_w_coeff_d, int width, int height)
{
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // Map 1D position to 2D grid
  unsigned int row = idx / width;
  unsigned int col = idx % width;

  if (idx < width * height)
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

  if (idx < width * height)
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

  if (idx < width * height)
  {
    unsigned int pos = row * width + col;

    pai_d[pos] = 10.1 * (band_nir_d[pos] - sqrt(band_red_d[pos])) + 3.1;

    if (pai_d[pos] < 0)
      pai_d[pos] = 0;
  }
}

__global__ void lai_kernel(float *reflectance_nir_d, float *reflectance_red_d, float *lai_d, int width, int height)
{
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // Map 1D position to 2D grid
  unsigned int row = idx / width;
  unsigned int col = idx % width;

  if (idx < width * height)
  {
    unsigned int pos = row * width + col;

    float savi = ((1 + 0.5) * (reflectance_nir_d[pos] - reflectance_red_d[pos])) / (0.5 + (reflectance_nir_d[pos] + reflectance_red_d[pos]));

    if (!isnan(savi) && savi > 0.687)
      lai_d[pos] = 6;
    if (!isnan(savi) && savi <= 0.687)
      lai_d[pos] = -log((0.69 - savi) / 0.59) / 0.91;
    if (!isnan(savi) && savi < 0.1)
      lai_d[pos] = 0;

    if (lai_d[pos] < 0)
      lai_d[pos] = 0;
  }
}

__global__ void evi_kernel(float *reflectance_nir_d, float *reflectance_red_d, float *reflectance_blue_d, float *evi_d, int width, int height)
{
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // Map 1D position to 2D grid
  unsigned int row = idx / width;
  unsigned int col = idx % width;

  if (idx < width * height)
  {
    unsigned int pos = row * width + col;

    evi_d[pos] = 2.5 * ((reflectance_nir_d[pos] - reflectance_red_d[pos]) / (reflectance_nir_d[pos] + (6 * reflectance_red_d[pos]) - (7.5 * reflectance_blue_d[pos]) + 1));

    if (evi_d[pos] < 0)
      evi_d[pos] = 0;
  }
}

__global__ void enb_kernel(float *lai_d, float *ndvi_d, float *enb_d, int width, int height)
{
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // Map 1D position to 2D grid
  unsigned int row = idx / width;
  unsigned int col = idx % width;

  if (idx < width * height)
  {
    unsigned int pos = row * width + col;

    if (lai_d[pos] == 0)
      enb_d[pos] = NAN;
    else
      enb_d[pos] = 0.97 + 0.0033 * lai_d[pos];

    if ((ndvi_d[pos] < 0) || (lai_d[pos] > 2.99))
      enb_d[pos] = 0.98;
  }
}

__global__ void eo_kernel(float *lai_d, float *ndvi_d, float *eo_d, int width, int height)
{
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // Map 1D position to 2D grid
  unsigned int row = idx / width;
  unsigned int col = idx % width;

  if (idx < width * height)
  {
    unsigned int pos = row * width + col;

    if (lai_d[pos] == 0)
      eo_d[pos] = NAN;
    else
      eo_d[pos] = 0.95 + 0.01 * lai_d[pos];

    if ((ndvi_d[pos] < 0) || (lai_d[pos] > 2.99))
      eo_d[pos] = 0.98;
  }
}

__global__ void ea_kernel(float *tal_d, float *ea_d, int width, int height)
{
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // Map 1D position to 2D grid
  unsigned int row = idx / width;
  unsigned int col = idx % width;

  if (idx < width * height)
  {
    unsigned int pos = row * width + col;
    ea_d[pos] = 0.85 * pow((-1 * log(tal_d[pos])), 0.09);
  }
}

__global__ void surface_temperature_kernel(float *enb_d, float *radiance_termal_d, float *surface_temperature_d, float k1, float k2, int width, int height)
{
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // Map 1D position to 2D grid
  unsigned int row = idx / width;
  unsigned int col = idx % width;

  if (idx < width * height)
  {
    unsigned int pos = row * width + col;
    surface_temperature_d[pos] = k2 / (log((enb_d[pos] * k1 / radiance_termal_d[pos]) + 1));

    if (surface_temperature_d[pos] < 0)
      surface_temperature_d[pos] = 0;
  }
}

__global__ void short_wave_radiation_kernel(float *tal_d, float *short_wave_radiation_d, float sun_elevation, float distance_earth_sun, float pi, int width, int height)
{
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // Map 1D position to 2D grid
  unsigned int row = idx / width;
  unsigned int col = idx % width;

  if (idx < width * height)
  {
    unsigned int pos = row * width + col;
    short_wave_radiation_d[pos] = (1367 * sin(sun_elevation * pi / 180) * tal_d[pos]) / (distance_earth_sun * distance_earth_sun);
  }
}

__global__ void large_wave_radiation_surface_kernel(float *surface_temperature_d, float *eo_d, float *large_wave_radiation_surface_d, int width, int height)
{
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // Map 1D position to 2D grid
  unsigned int row = idx / width;
  unsigned int col = idx % width;

  if (idx < width * height)
  {
    unsigned int pos = row * width + col;
    float temperature_pixel = surface_temperature_d[pos];
    float surface_temperature_pow_4 = temperature_pixel * temperature_pixel * temperature_pixel * temperature_pixel;
    large_wave_radiation_surface_d[pos] = eo_d[pos] * 5.67 * 1e-8 * surface_temperature_pow_4;
  }
}

__global__ void large_wave_radiation_atmosphere_kernel(float *ea_d, float *large_wave_radiation_atmosphere_d, float temperature, int width, int height)
{
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // Map 1D position to 2D grid
  unsigned int row = idx / width;
  unsigned int col = idx % width;

  float temperature_kelvin = temperature + 273.15;
  float temperature_kelvin_pow_4 = temperature_kelvin * temperature_kelvin * temperature_kelvin * temperature_kelvin;

  if (idx < width * height)
  {
    unsigned int pos = row * width + col;
    large_wave_radiation_atmosphere_d[pos] = ea_d[pos] * 5.67 * 1e-8 * temperature_kelvin_pow_4;
  }
}

__global__ void net_radiation_kernel(float *short_wave_radiation_d, float *albedo_d, float *large_wave_radiation_atmosphere_d, float *large_wave_radiation_surface_d, float *eo_d, float *net_radiation_d, int width, int height)
{
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // Map 1D position to 2D grid
  unsigned int row = idx / width;
  unsigned int col = idx % width;

  if (idx < width * height)
  {
    unsigned int pos = row * width + col;
    net_radiation_d[pos] = short_wave_radiation_d[pos] - (short_wave_radiation_d[pos] * albedo_d[pos]) + large_wave_radiation_atmosphere_d[pos] - large_wave_radiation_surface_d[pos] - (1 - eo_d[pos]) * large_wave_radiation_atmosphere_d[pos];

    if (net_radiation_d[pos] < 0)
      net_radiation_d[pos] = 0;
  }
}

__global__ void soil_heat_kernel(float *ndvi_d, float *albedo_d, float *surface_temperature_d, float *net_radiation_d, float *soil_heat_d, int width, int height)
{
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // Map 1D position to 2D grid
  unsigned int row = idx / width;
  unsigned int col = idx % width;

  if (idx < width * height)
  {
    unsigned int pos = row * width + col;

    if ((ndvi_d[pos] < 0) || (ndvi_d[pos] > 0))
    {
      float ndvi_pixel_pow_4 = ndvi_d[pos] * ndvi_d[pos] * ndvi_d[pos] * ndvi_d[pos];
      soil_heat_d[pos] = (surface_temperature_d[pos] - 273.15) * (0.0038 + 0.0074 * albedo_d[pos]) * (1 - 0.98 * ndvi_pixel_pow_4) * net_radiation_d[pos];
    }
    else
      soil_heat_d[pos] = 0.5 * net_radiation_d[pos];

    if (soil_heat_d[pos] < 0)
      soil_heat_d[pos] = 0;
  }
}

__global__ void d0_kernel(float *pai_d, float *d0_d, float CD1, float HGHT, int width, int height)
{
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // Map 1D position to 2D grid
  unsigned int row = idx / width;
  unsigned int col = idx % width;

  if (idx < width * height)
  {
    unsigned int pos = row * width + col;
    float cd1_pai_root = sqrt(CD1 * pai_d[pos]);

    float DISP = HGHT * ((1 - (1 / cd1_pai_root)) + (pow(exp(1.0), -cd1_pai_root) / cd1_pai_root));
    if (pai_d[pos] < 0)
    {
      DISP = 0;
    }

    d0_d[pos] = DISP;
  }
}

__global__ void ustar_kernel(float *zom_d, float *d0_d, float *ustar_d, float u10, int width, int height)
{
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // Map 1D position to 2D grid
  unsigned int row = idx / width;
  unsigned int col = idx % width;

  float zu = 10;
  if (idx < width * height)
  {
    unsigned int pos = row * width + col;
    ustar_d[pos] = (u10 * VON_KARMAN) / log((zu - d0_d[pos]) / zom_d[pos]);
  }
}

__global__ void zom_kernel(float *d0_d, float *pai_d, float *zom_d, float A_ZOM, float B_ZOM, int width, int height)
{
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // Map 1D position to 2D grid
  unsigned int row = idx / width;
  unsigned int col = idx % width;

  float HGHT = 4;
  float CD = 0.01;
  float CR = 0.35;
  float PSICORR = 0.2;

  if (idx < width * height)
  {
    unsigned int pos = row * width + col;

    float gama = pow((CD + CR * (pai_d[pos] / 2)), -0.5);

    if (gama < 3.3)
      gama = 3.3;

    zom_d[pos] = (HGHT - d0_d[pos]) * pow(exp(1.0), (-VON_KARMAN * gama) + PSICORR);
  }
}

__global__ void kb_kernel(float *zom_d, float *ustar_d, float *pai_d, float *kb1_d, float *ndvi_d, int width, int height, float ndvi_max, float ndvi_min)
{
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // Map 1D position to 2D grid
  unsigned int row = idx / width;
  unsigned int col = idx % width;

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

  if (idx < width * height)
  {
    unsigned int pos = row * width + col;

    float Re_star = (ustar_d[pos] * 0.009) / visc;
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

__global__ void aerodynamic_resistance_kernel(float *zom_d, float *d0_d, float *ustar_d, float *kb1_d, float *aerodynamic_resistance_d, int width, int height)
{
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // Map 1D position to 2D grid
  unsigned int row = idx / width;
  unsigned int col = idx % width;

  float zu = 10.0;

  if (idx < width * height)
  {
    unsigned int pos = row * width + col;

    float DISP = d0_d[pos];
    float zom = zom_d[pos];
    float zoh_terra = zom / pow(exp(1.0), (kb1_d[pos]));

    float temp_kb_1_terra = log(zom / zoh_terra);
    float temp_rah1_terra = (1 / (ustar_d[pos] * VON_KARMAN));
    float temp_rah2 = log(((zu - DISP) / zom));
    float temp_rah3_terra = temp_rah1_terra * temp_kb_1_terra;

    aerodynamic_resistance_d[pos] = temp_rah1_terra * temp_rah2 + temp_rah3_terra;
  }
}

__global__ void sensible_heat_flux_kernel(Candidate *d_hotCandidates, Candidate *d_coldCandidates, int hot_pos, int cold_pos, float *surface_temperature_d,
                                          float *rah_d, float *net_radiation_d, float *soil_heat_d, float *sensible_heat_flux_d, int width, int height)
{
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // Map 1D position to 2D grid
  unsigned int row = idx / width;
  unsigned int col = idx % width;

  if (idx < width * height)
  {
    unsigned int pos = row * width + col;

    Candidate hot_pixel = d_hotCandidates[hot_pos];
    Candidate cold_pixel = d_coldCandidates[cold_pos];

    float dt_pq_terra = H_pq_terra * rah_ini_pq_terra / (RHO * SPECIFIC_HEAT_AIR);
    float dt_pf_terra = H_pf_terra * rah_ini_pf_terra / (RHO * SPECIFIC_HEAT_AIR);

    float b = (dt_pq_terra - dt_pf_terra) / (hot_pixel.temperature - cold_pixel.temperature);
    float a = dt_pf_terra - (b * (cold_pixel.temperature - 273.15));

    sensible_heat_flux_d[pos] = RHO * SPECIFIC_HEAT_AIR * (a + b * (surface_temperature_d[pos] - 273.15)) / rah_d[pos];

    if (!isnan(sensible_heat_flux_d[pos]) && sensible_heat_flux_d[pos] > (net_radiation_d[pos] - soil_heat_d[pos]))
    {
      sensible_heat_flux_d[pos] = net_radiation_d[pos] - soil_heat_d[pos];
    }
  }
}

__global__ void latent_heat_flux_kernel(float *net_radiation_d, float *soil_heat_d, float *sensible_heat_flux_d, float *latent_heat_flux_d, int width, int height)
{
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // Map 1D position to 2D grid
  unsigned int row = idx / width;
  unsigned int col = idx % width;

  if (idx < width * height)
  {
    unsigned int pos = row * width + col;
    latent_heat_flux_d[pos] = net_radiation_d[pos] - soil_heat_d[pos] - sensible_heat_flux_d[pos];
  }
}

__global__ void net_radiation_24h_kernel(float *albedo_d, float Rs24h, float Ra24h, float *net_radiation_24h_d, int width, int height)
{
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // Map 1D position to 2D grid
  unsigned int row = idx / width;
  unsigned int col = idx % width;

  int FL = 110;

  if (idx < width * height)
  {
    unsigned int pos = row * width + col;
    net_radiation_24h_d[pos] = (1 - albedo_d[pos]) * Rs24h - FL * Rs24h / Ra24h;
  }
}

__global__ void evapotranspiration_fraction_kernel(float *net_radiation_d, float *soil_heat_d, float *latent_heat_flux_d, float *evapotranspiration_fraction_d, int width, int height)
{
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // Map 1D position to 2D grid
  unsigned int row = idx / width;
  unsigned int col = idx % width;

  if (idx < width * height)
  {
    unsigned int pos = row * width + col;
    evapotranspiration_fraction_d[pos] = latent_heat_flux_d[pos] / (net_radiation_d[pos] - soil_heat_d[pos]);
  }
}

__global__ void sensible_heat_flux_24h_kernel(float *net_radiation_24h_d, float *evapotranspiration_fraction_d, float *sensible_heat_flux_24h_d, int width, int height)
{
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // Map 1D position to 2D grid
  unsigned int row = idx / width;
  unsigned int col = idx % width;

  if (idx < width * height)
  {
    unsigned int pos = row * width + col;
    sensible_heat_flux_24h_d[pos] = (1 - evapotranspiration_fraction_d[pos]) * net_radiation_24h_d[pos];
  }
}

__global__ void latent_heat_flux_24h_kernel(float *net_radiation_24h_d, float *evapotranspiration_fraction_d, float *latent_heat_flux_24h_d, int width, int height)
{
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // Map 1D position to 2D grid
  unsigned int row = idx / width;
  unsigned int col = idx % width;

  if (idx < width * height)
  {
    unsigned int pos = row * width + col;
    latent_heat_flux_24h_d[pos] = evapotranspiration_fraction_d[pos] * net_radiation_24h_d[pos];
  }
}

__global__ void evapotranspiration_24h_kernel(float *latent_heat_flux_24h_d, float *evapotranspiration_24h_d, float v7_max, float v7_min, int width, int height)
{
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // Map 1D position to 2D grid
  unsigned int row = idx / width;
  unsigned int col = idx % width;

  if (idx < width * height)
  {
    unsigned int pos = row * width + col;
    evapotranspiration_24h_d[pos] = (latent_heat_flux_24h_d[pos] * 86400) / ((2.501 - 0.00236 * (v7_max + v7_min) / 2) * 1e+6);
  }
}

__global__ void evapotranspiration_kernel(float *net_radiation_24h_d, float *evapotranspiration_fraction_d, float *evapotranspiration_d, int width, int height)
{
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // Map 1D position to 2D grid
  unsigned int row = idx / width;
  unsigned int col = idx % width;

  if (idx < width * height)
  {
    unsigned int pos = row * width + col;
    evapotranspiration_d[pos] = net_radiation_24h_d[pos] * evapotranspiration_fraction_d[pos] * 0.035;
  }
}

__global__ void rah_correction_cycle_STEEP(Candidate *d_hotCandidates, Candidate *d_coldCandidates, int hot_idx, int cold_idx,
                                           float *ndvi_pointer, float *surf_temp_pointer, float *d0_pointer, float *kb1_pointer,
                                           float *zom_pointer, float *ustar_pointer, float *rah_pointer, float *H_pointer,
                                           float ndvi_max, float ndvi_min, int height, int width)
{
  // Identify 1D position
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // Map 1D position to 2D grid
  unsigned int row = idx / width;
  unsigned int col = idx % width;

  if (idx < width * height)
  {
    unsigned int pos = row * width + col;

    Candidate hot_pixel = d_hotCandidates[hot_idx];
    Candidate cold_pixel = d_coldCandidates[cold_idx];
    unsigned int hot_pos = hot_pixel.line * width + hot_pixel.col;
    unsigned int cold_pos = cold_pixel.line * width + cold_pixel.col;

    float fc_hot = 1 - pow((ndvi_pointer[hot_pos] - ndvi_max) / (ndvi_min - ndvi_max), 0.4631);
    float fc_cold = 1 - pow((ndvi_pointer[cold_pos] - ndvi_max) / (ndvi_min - ndvi_max), 0.4631);

    rah_ini_pq_terra = rah_pointer[hot_pos];
    rah_ini_pf_terra = rah_pointer[cold_pos];

    float LEc_terra = 0.55 * fc_hot * (hot_pixel.net_radiation - hot_pixel.soil_heat_flux) * 0.78;
    float LEc_terra_pf = 1.75 * fc_cold * (cold_pixel.net_radiation - cold_pixel.soil_heat_flux) * 0.78;

    H_pf_terra = cold_pixel.net_radiation - cold_pixel.soil_heat_flux - LEc_terra_pf;
    float dt_pf_terra = H_pf_terra * rah_ini_pf_terra / (RHO * SPECIFIC_HEAT_AIR);

    H_pq_terra = hot_pixel.net_radiation - hot_pixel.soil_heat_flux - LEc_terra;
    float dt_pq_terra = H_pq_terra * rah_ini_pq_terra / (RHO * SPECIFIC_HEAT_AIR);

    float b = (dt_pq_terra - dt_pf_terra) / (hot_pixel.temperature - cold_pixel.temperature);
    float a = dt_pf_terra - (b * (cold_pixel.temperature - 273.15));

    float DISP = d0_pointer[pos];
    float dT_ini_terra = a + b * (surf_temp_pointer[pos] - 273.15);

    float sensibleHeatFlux = RHO * SPECIFIC_HEAT_AIR * (dT_ini_terra) / rah_pointer[pos];
    float L = -1 * ((RHO * SPECIFIC_HEAT_AIR * pow(ustar_pointer[pos], 3) * surf_temp_pointer[pos]) / (VON_KARMAN * GRAVITY * sensibleHeatFlux));

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

    float ust = (VON_KARMAN * ustar_pointer[pos]) / (log((10 - DISP) / zom_pointer[pos]) - psi200);

    float zoh_terra = zom_pointer[pos] / pow(exp(1.0), (kb1_pointer[pos]));
    float temp_rah1_corr_terra = (ust * VON_KARMAN);
    float temp_rah2_corr_terra = log((10 - DISP) / zom_pointer[pos]) - psi2;
    float temp_rah3_corr_terra = temp_rah1_corr_terra * log(zom_pointer[pos] / zoh_terra);
    float rah = (temp_rah1_corr_terra * temp_rah2_corr_terra) + temp_rah3_corr_terra;

    ustar_pointer[pos] = ust;
    rah_pointer[pos] = rah;
    H_pointer[pos] = sensibleHeatFlux;
  }
}

__global__ void rah_correction_cycle_ASEBAL(Candidate *d_hotCandidates, Candidate *d_coldCandidates, int hot_pos, int cold_pos,
                                            float *ndvi_pointer, float *surf_temp_pointer, float *kb1_pointer, float *zom_pointer,
                                            float *ustar_pointer, float *rah_pointer, float *H_pointer, float ndvi_max, float ndvi_min,
                                            float u200, int height, int width, int *stop_condition)
{
  // Identify 1D position
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // Map 1D position to 2D grid
  unsigned int row = idx / width;
  unsigned int col = idx % width;

  if (idx < width * height)
  {
    unsigned int pos = row * width + col;

    Candidate hot_pixel = d_hotCandidates[hot_pos];
    Candidate cold_pixel = d_coldCandidates[cold_pos];
    unsigned int hot_pos = hot_pixel.line * width + hot_pixel.col;
    unsigned int cold_pos = cold_pixel.line * width + cold_pixel.col;

    float fc_hot = 1 - pow((ndvi_pointer[hot_pos] - ndvi_max) / (ndvi_min - ndvi_max), 0.4631);
    float fc_cold = 1 - pow((ndvi_pointer[cold_pos] - ndvi_max) / (ndvi_min - ndvi_max), 0.4631);

    rah_ini_pq_terra = rah_pointer[hot_pos];
    rah_ini_pf_terra = rah_pointer[cold_pos];

    float LEc_terra = 0.55 * fc_hot * (hot_pixel.net_radiation - hot_pixel.soil_heat_flux) * 0.78;
    float LEc_terra_pf = 1.75 * fc_cold * (cold_pixel.net_radiation - cold_pixel.soil_heat_flux) * 0.78;

    H_pf_terra = cold_pixel.net_radiation - cold_pixel.soil_heat_flux - LEc_terra_pf;
    float dt_pf_terra = H_pf_terra * rah_ini_pf_terra / (RHO * SPECIFIC_HEAT_AIR);

    H_pq_terra = hot_pixel.net_radiation - hot_pixel.soil_heat_flux - LEc_terra;
    float dt_pq_terra = H_pq_terra * rah_ini_pq_terra / (RHO * SPECIFIC_HEAT_AIR);

    float b = (dt_pq_terra - dt_pf_terra) / (surf_temp_pointer[hot_pos] - surf_temp_pointer[cold_pos]);
    float a = dt_pf_terra - (b * (surf_temp_pointer[cold_pos] - 273.15));

    float dT_ini_terra = a + b * (surf_temp_pointer[pos] - 273.15);

    float sensibleHeatFlux = RHO * SPECIFIC_HEAT_AIR * (dT_ini_terra) / rah_pointer[pos];
    float L = -1 * ((RHO * SPECIFIC_HEAT_AIR * pow(ustar_pointer[pos], 3) * surf_temp_pointer[pos]) / (VON_KARMAN * GRAVITY * sensibleHeatFlux));

    float y1 = pow((1 - (16 * 0.1) / L), 0.25);
    float y2 = pow((1 - (16 * 2) / L), 0.25);
    float x200 = pow((1 - (16 * 200) / L), 0.25);

    float psi1, psi2, psi200;
    if (!isnan(L) && L > 0)
    {
      psi1 = -5 * (0.1 / L);
      psi2 = -5 * (2 / L);
      psi200 = -5 * (2 / L);
    }
    else
    {
      psi1 = 2 * log((1 + y1 * y1) / 2);
      psi2 = 2 * log((1 + y2 * y2) / 2);
      psi200 = 2 * log((1 + x200) / 2) + log((1 + x200 * x200) / 2) - 2 * atan(x200) + 0.5 * M_PI;
    }

    float ust = (VON_KARMAN * u200) / (log(200 / zom_pointer[pos]) - psi200);
    float rah = (log(2 / 0.1) - psi2 + psi1) / (ustar_pointer[pos] * VON_KARMAN);

    ustar_pointer[pos] = ust;
    rah_pointer[pos] = rah;
    H_pointer[pos] = sensibleHeatFlux;

    if ((pos == hot_pos) && (fabs(1 - rah_ini_pq_terra / rah_pointer[hot_pos]) < 0.05))
    {
      atomicExch(stop_condition, 1); // Set the global flag if any thread meets the condition
    }
  }
}

__global__ void filter_valid_values(const float *target, float *filtered, int height_band, int width_band, int *pos)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int size = height_band * width_band;

  if (idx < size)
  {
    float value = target[idx];
    if (!isnan(value) && !isinf(value))
    {
      int position = atomicAdd(pos, 1);
      filtered[position] = value;
    }
  }
}

__global__ void process_pixels_STEEP(Candidate *hotCandidates, Candidate *coldCandidates, int *d_indexes,
                                     float *ndvi, float *surface_temperature, float *albedo, float *net_radiation, float *soil_heat, float *ho,
                                     float ndviQuartileLow, float ndviQuartileHigh, float tsQuartileLow, float tsQuartileMid, float tsQuartileHigh,
                                     float albedoQuartileLow, float albedoQuartileMid, float albedoQuartileHigh, int height_band, int width_band)
{
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // Map 1D position to 2D grid
  unsigned int row = idx / width_band;
  unsigned int col = idx % width_band;

  if (idx < width_band * height_band)
  {
    unsigned int pos = row * width_band + col;

    ho[pos] = net_radiation[pos] - soil_heat[pos];

    bool hotNDVI = !isnan(ndvi[pos]) && ndvi[pos] > 0.10 && ndvi[pos] < ndviQuartileLow;
    bool hotAlbedo = !isnan(albedo[pos]) && albedo[pos] > albedoQuartileMid && albedo[pos] < albedoQuartileHigh;
    bool hotTS = !isnan(surface_temperature[pos]) && surface_temperature[pos] > tsQuartileMid && surface_temperature[pos] < tsQuartileHigh;

    bool coldNDVI = !isnan(ndvi[pos]) && ndvi[pos] > ndviQuartileHigh;
    bool coldAlbedo = !isnan(surface_temperature[pos]) && albedo[pos] > albedoQuartileLow && albedo[pos] < albedoQuartileMid;
    bool coldTS = !isnan(albedo[pos]) && surface_temperature[pos] < tsQuartileLow;

    if (hotAlbedo && hotNDVI && hotTS)
    {
      int ih = atomicAdd(&d_indexes[0], 1);
      hotCandidates[ih] = Candidate(ndvi[pos], surface_temperature[pos], net_radiation[pos], soil_heat[pos], ho[pos], row, col);
    }

    if (coldNDVI && coldAlbedo && coldTS)
    {
      int ic = atomicAdd(&d_indexes[1], 1);
      coldCandidates[ic] = Candidate(ndvi[pos], surface_temperature[pos], net_radiation[pos], soil_heat[pos], ho[pos], row, col);
    }
  }
}

__global__ void process_pixels_ASEBAL(Candidate *hotCandidates, Candidate *coldCandidates, int *d_indexes,
                                      float *ndvi, float *surface_temperature, float *albedo, float *net_radiation, float *soil_heat, float *ho,
                                      float ndviHOTQuartile, float ndviCOLDQuartile, float tsHOTQuartile, float tsCOLDQuartile,
                                      float albedoHOTQuartile, float albedoCOLDQuartile, int height_band, int width_band)
{
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // Map 1D position to 2D grid
  unsigned int row = idx / width_band;
  unsigned int col = idx % width_band;

  if (idx < width_band * height_band)
  {
    unsigned int pos = row * width_band + col;

    ho[pos] = net_radiation[pos] - soil_heat[pos];

    bool hotNDVI = !isnan(ndvi[pos]) && ndvi[pos] > 0.10 && ndvi[pos] < ndviHOTQuartile;
    bool hotAlbedo = !isnan(albedo[pos]) && albedo[pos] > albedoHOTQuartile;
    bool hotTS = !isnan(surface_temperature[pos]) && surface_temperature[pos] > tsHOTQuartile;

    bool coldNDVI = !isnan(ndvi[pos]) && ndvi[pos] > ndviCOLDQuartile;
    bool coldAlbedo = !isnan(surface_temperature[pos]) && albedo[pos] < albedoCOLDQuartile;
    bool coldTS = !isnan(albedo[pos]) && surface_temperature[pos] < tsCOLDQuartile;

    if (hotAlbedo && hotNDVI && hotTS)
    {
      int ih = atomicAdd(&d_indexes[0], 1);
      hotCandidates[ih] = Candidate(ndvi[pos], surface_temperature[pos], net_radiation[pos], soil_heat[pos], ho[pos], row, col);
    }

    if (coldNDVI && coldAlbedo && coldTS)
    {
      int ic = atomicAdd(&d_indexes[1], 1);
      coldCandidates[ic] = Candidate(ndvi[pos], surface_temperature[pos], net_radiation[pos], soil_heat[pos], ho[pos], row, col);
    }
  }
}
