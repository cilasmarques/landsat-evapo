#include "kernels.cuh"

__global__ void rad_kernel(float *band1_d, float *band2_d, float *band3_d, float *band4_d,
                           float *band5_d, float *band6_d, float *band7_d,
                           float *radiance1_d, float *radiance2_d, float *radiance3_d, float *radiance4_d,
                           float *radiance5_d, float *radiance6_d, float *radiance7_d,
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

    radiance1_d[pos] = band1_d[pos] * grenscale1_d + brescale1_d;
    radiance2_d[pos] = band2_d[pos] * grenscale2_d + brescale2_d;
    radiance3_d[pos] = band3_d[pos] * grenscale3_d + brescale3_d;
    radiance4_d[pos] = band4_d[pos] * grenscale4_d + brescale4_d;
    radiance5_d[pos] = band5_d[pos] * grenscale5_d + brescale5_d;
    radiance6_d[pos] = band6_d[pos] * grenscale6_d + brescale6_d;
    radiance7_d[pos] = band7_d[pos] * grenscale7_d + brescale7_d;
  }
}

__global__ void ref_kernel(float sin_sun,
                           float *radiance1_d, float *radiance2_d, float *radiance3_d, float *radiance4_d,
                           float *radiance5_d, float *radiance6_d, float *radiance7_d,
                           float *reflectance1_d, float *reflectance2_d, float *reflectance3_d, float *reflectance4_d,
                           float *reflectance5_d, float *reflectance6_d, float *reflectance7_d,
                           int width, int height)
{
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // Map 1D position to 2D grid
  unsigned int row = idx / width;
  unsigned int col = idx % width;

  if (col < width && row < height)
  {
    unsigned int pos = row * width + col;

    reflectance1_d[pos] = radiance1_d[pos] / sin_sun;
    reflectance2_d[pos] = radiance2_d[pos] / sin_sun;
    reflectance3_d[pos] = radiance3_d[pos] / sin_sun;
    reflectance4_d[pos] = radiance4_d[pos] / sin_sun;
    reflectance5_d[pos] = radiance5_d[pos] / sin_sun;
    reflectance6_d[pos] = radiance6_d[pos] / sin_sun;
    reflectance7_d[pos] = radiance7_d[pos] / sin_sun;
  }
}

__global__ void ref_kernel(double PI, float sin_sun, float distance_earth_sun,
                           float esun1, float esun2, float esun3, float esun4,
                           float esun5, float esun6, float esun7,
                           float *radiance1_d, float *radiance2_d, float *radiance3_d, float *radiance4_d,
                           float *radiance5_d, float *radiance6_d, float *radiance7_d,
                           float *reflectance1_d, float *reflectance2_d, float *reflectance3_d, float *reflectance4_d,
                           float *reflectance5_d, float *reflectance6_d, float *reflectance7_d,
                           int width, int height)
{
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // Map 1D position to 2D grid
  unsigned int row = idx / width;
  unsigned int col = idx % width;

  if (col < width && row < height)
  {
    unsigned int pos = row * width + col;

    reflectance1_d[pos] = (PI * radiance1_d[pos] * distance_earth_sun * distance_earth_sun) / (esun1 * sin_sun);
    reflectance2_d[pos] = (PI * radiance2_d[pos] * distance_earth_sun * distance_earth_sun) / (esun2 * sin_sun);
    reflectance3_d[pos] = (PI * radiance3_d[pos] * distance_earth_sun * distance_earth_sun) / (esun3 * sin_sun);
    reflectance4_d[pos] = (PI * radiance4_d[pos] * distance_earth_sun * distance_earth_sun) / (esun4 * sin_sun);
    reflectance5_d[pos] = (PI * radiance5_d[pos] * distance_earth_sun * distance_earth_sun) / (esun5 * sin_sun);
    reflectance6_d[pos] = (PI * radiance6_d[pos] * distance_earth_sun * distance_earth_sun) / (esun6 * sin_sun);
    reflectance7_d[pos] = (PI * radiance7_d[pos] * distance_earth_sun * distance_earth_sun) / (esun7 * sin_sun);
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
