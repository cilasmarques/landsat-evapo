#include "kernels.cuh"

__global__ void rad_kernel(float *band_blue_d, float *band_green_d, float *band_red_d, float *band_nir_d,
                           float *band_swir1_d, float *band_termal_d, float *band_swir2_d,
                           float *radiance_blue_d, float *radiance_green_d, float *radiance_red_d, float *radiance_nir_d,
                           float *radiance_swir1_d, float *radiance_termal_d, float *radiance_swir2_d,
                           float grenscale1_d, float brescale1_d,
                           float grenscale2_d, float brescale2_d,
                           float grenscale3_d, float brescale3_d,
                           float grenscale4_d, float brescale4_d,
                           float grenscale5_d, float brescale5_d,
                           float grenscale6_d, float brescale6_d,
                           float grenscale7_d, float brescale7_d,
                           float grenscale8_d,
                           int width, int height)
{
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // Map 1D position to 2D grid
  unsigned int row = idx / width;
  unsigned int col = idx % width;

  if (col < width && row < height)
  {
    unsigned int pos = row * width + col;

    radiance_blue_d[pos] = band_blue_d[pos] * grenscale1_d + brescale1_d;
    radiance_green_d[pos] = band_green_d[pos] * grenscale2_d + brescale2_d;
    radiance_red_d[pos] = band_red_d[pos] * grenscale3_d + brescale3_d;
    radiance_nir_d[pos] = band_nir_d[pos] * grenscale4_d + brescale4_d;
    radiance_swir1_d[pos] = band_swir1_d[pos] * grenscale5_d + brescale5_d;
    radiance_termal_d[pos] = band_termal_d[pos] * grenscale6_d + brescale6_d;
    radiance_swir2_d[pos] = band_swir2_d[pos] * grenscale7_d + brescale7_d;
  }
}

__global__ void ref_kernel(float sin_sun,
                           float *radiance_blue_d, float *radiance_green_d, float *radiance_red_d, float *radiance_nir_d,
                           float *radiance_swir1_d, float *radiance_termal_d, float *radiance_swir2_d,
                           float *reflectance_blue_d, float *reflectance_green_d, float *reflectance_red_d, float *reflectance_nir_d,
                           float *reflectance_swir1_d, float *reflectance_termal_d, float *reflectance_swir2_d,
                           int width, int height)
{
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // Map 1D position to 2D grid
  unsigned int row = idx / width;
  unsigned int col = idx % width;

  if (col < width && row < height)
  {
    unsigned int pos = row * width + col;

    reflectance_blue_d[pos] = radiance_blue_d[pos] / sin_sun;
    reflectance_green_d[pos] = radiance_green_d[pos] / sin_sun;
    reflectance_red_d[pos] = radiance_red_d[pos] / sin_sun;
    reflectance_nir_d[pos] = radiance_nir_d[pos] / sin_sun;
    reflectance_swir1_d[pos] = radiance_swir1_d[pos] / sin_sun;
    reflectance_termal_d[pos] = radiance_termal_d[pos] / sin_sun;
    reflectance_swir2_d[pos] = radiance_swir2_d[pos] / sin_sun;
  }
}

__global__ void ref_kernel(double PI, float sin_sun, float distance_earth_sun,
                           float esun1, float esun2, float esun3, float esun4,
                           float esun5, float esun6, float esun7,
                           float *radiance_blue_d, float *radiance_green_d, float *radiance_red_d, float *radiance_nir_d,
                           float *radiance_swir1_d, float *radiance_termal_d, float *radiance_swir2_d,
                           float *reflectance_blue_d, float *reflectance_green_d, float *reflectance_red_d, float *reflectance_nir_d,
                           float *reflectance_swir1_d, float *reflectance_termal_d, float *reflectance_swir2_d,
                           int width, int height)
{
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // Map 1D position to 2D grid
  unsigned int row = idx / width;
  unsigned int col = idx % width;

  if (col < width && row < height)
  {
    unsigned int pos = row * width + col;

    reflectance_blue_d[pos] = (PI * radiance_blue_d[pos] * distance_earth_sun * distance_earth_sun) / (esun1 * sin_sun);
    reflectance_green_d[pos] = (PI * radiance_green_d[pos] * distance_earth_sun * distance_earth_sun) / (esun2 * sin_sun);
    reflectance_red_d[pos] = (PI * radiance_red_d[pos] * distance_earth_sun * distance_earth_sun) / (esun3 * sin_sun);
    reflectance_nir_d[pos] = (PI * radiance_nir_d[pos] * distance_earth_sun * distance_earth_sun) / (esun4 * sin_sun);
    reflectance_swir1_d[pos] = (PI * radiance_swir1_d[pos] * distance_earth_sun * distance_earth_sun) / (esun5 * sin_sun);
    reflectance_termal_d[pos] = (PI * radiance_termal_d[pos] * distance_earth_sun * distance_earth_sun) / (esun6 * sin_sun);
    reflectance_swir2_d[pos] = (PI * radiance_swir2_d[pos] * distance_earth_sun * distance_earth_sun) / (esun7 * sin_sun);
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
