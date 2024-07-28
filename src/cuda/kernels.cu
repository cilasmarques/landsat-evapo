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

__global__ void rah_correction_cycle_STEEP(float *surface_temperature_pointer, float *d0_pointer, float *kb1_pointer, float *zom_pointer, float *ustarR_pointer,
                                           float *ustarW_pointer, float *rahR_pointer, float *rahW_pointer, float *H_pointer, double a, double b, int height,
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

    double DISP = d0_pointer[pos];
    double dT_ini_terra = a + b * (surface_temperature_pointer[pos] - 273.15);

    double sensibleHeatFlux = RHO * SPECIFIC_HEAT_AIR * (dT_ini_terra) / rahR_pointer[pos];
    double L = -1 * ((RHO * SPECIFIC_HEAT_AIR * pow(ustarR_pointer[pos], 3) * surface_temperature_pointer[pos]) / (VON_KARMAN * GRAVITY * sensibleHeatFlux));

    double y2 = pow((1 - (16 * (10 - DISP)) / L), 0.25);
    double x200 = pow((1 - (16 * (10 - DISP)) / L), 0.25);

    double psi2, psi200;
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

    double ust = (VON_KARMAN * ustarR_pointer[pos]) / (log((10 - DISP) / zom_pointer[pos]) - psi200);

    double zoh_terra = zom_pointer[pos] / pow(exp(1.0), (kb1_pointer[pos]));
    double temp_rah1_corr_terra = (ust * VON_KARMAN);
    double temp_rah2_corr_terra = log((10 - DISP) / zom_pointer[pos]) - psi2;
    double temp_rah3_corr_terra = temp_rah1_corr_terra * log(zom_pointer[pos] / zoh_terra);
    double rah = (temp_rah1_corr_terra * temp_rah2_corr_terra) + temp_rah3_corr_terra;

    ustarW_pointer[pos] = ust;
    rahW_pointer[pos] = rah;
    H_pointer[pos] = sensibleHeatFlux;
  }
}
