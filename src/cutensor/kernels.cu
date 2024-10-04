#include "kernels.cuh"

__global__ void invalid_rad_ref_kernel(float *albedo, float *radiance_blue_d, float *radiance_green_d, float *radiance_red_d, float *radiance_nir_d, float *radiance_swir1_d, float *radiance_termal_d, float *radiance_swir2_d, float *reflectance_blue_d, float *reflectance_green_d, float *reflectance_red_d, float *reflectance_nir_d, float *reflectance_swir1_d, float *reflectance_termal_d, float *reflectance_swir2_d, int width, int height)
{
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // Map 1D position to 2D grid
  unsigned int row = idx / width;
  unsigned int col = idx % width;

  if (idx < width * height)
  {
    unsigned int pos = row * width + col;

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

    if (albedo[pos] <= 0)
      albedo[pos] = NAN;
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

__global__ void rah_correction_cycle_STEEP(float *surface_temperature_pointer, float *d0_pointer, float *kb1_pointer, float *zom_pointer,
                                           float *ustar_pointer, float *rah_pointer, float *H_pointer, float a, float b, int height,
                                           int width)
{
  // Identify 1D position
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // Map 1D position to 2D grid
  unsigned int row = idx / width;
  unsigned int col = idx % width;

  if (idx < width * height)
  {
    unsigned int pos = row * width + col;

    float DISP = d0_pointer[pos];
    float dT_ini_terra = a + b * (surface_temperature_pointer[pos] - 273.15);

    float sensibleHeatFlux = RHO * SPECIFIC_HEAT_AIR * (dT_ini_terra) / rah_pointer[pos];
    float L = -1 * ((RHO * SPECIFIC_HEAT_AIR * pow(ustar_pointer[pos], 3) * surface_temperature_pointer[pos]) / (VON_KARMAN * GRAVITY * sensibleHeatFlux));

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

__global__ void rah_correction_cycle_ASEBAL(float *surface_temperature_pointer, float *kb1_pointer, float *zom_pointer, float *ustar_pointer, 
                                            float *rah_pointer, float *H_pointer, float a, float b, float u200, int height, int width)
{
  // Identify 1D position
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // Map 1D position to 2D grid
  unsigned int row = idx / width;
  unsigned int col = idx % width;

  if (idx < width * height)
  {
    unsigned int pos = row * width + col;

    float dT_ini_terra = a + b * (surface_temperature_pointer[pos] - 273.15);

    float sensibleHeatFlux = RHO * SPECIFIC_HEAT_AIR * (dT_ini_terra) / rah_pointer[pos];
    float L = -1 * ((RHO * SPECIFIC_HEAT_AIR * pow(ustar_pointer[pos], 3) * surface_temperature_pointer[pos]) / (VON_KARMAN * GRAVITY * sensibleHeatFlux));

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
  }
}

// __global__ void invalid_rad_kernel(float *radiance_blue_d, float *radiance_green_d, float *radiance_red_d, float *radiance_nir_d, float *radiance_swir1_d, float *radiance_termal_d, float *radiance_swir2_d, int width, int height)
// {
//   unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

//   // Map 1D position to 2D grid
//   unsigned int row = idx / width;
//   unsigned int col = idx % width;

//   if (idx < width * height)
//   {
//     unsigned int pos = row * width + col;

//     if (radiance_blue_d[pos] <= 0)
//       radiance_blue_d[pos] = NAN;
//     if (radiance_green_d[pos] <= 0)
//       radiance_green_d[pos] = NAN;
//     if (radiance_red_d[pos] <= 0)
//       radiance_red_d[pos] = NAN;
//     if (radiance_nir_d[pos] <= 0)
//       radiance_nir_d[pos] = NAN;
//     if (radiance_swir1_d[pos] <= 0)
//       radiance_swir1_d[pos] = NAN;
//     if (radiance_termal_d[pos] <= 0)
//       radiance_termal_d[pos] = NAN;
//     if (radiance_swir2_d[pos] <= 0)
//       radiance_swir2_d[pos] = NAN;
//   }
// }

// __global__ void invalid_ref_kernel(float *reflectance_blue_d, float *reflectance_green_d, float *reflectance_red_d, float *reflectance_nir_d, float *reflectance_swir1_d, float *reflectance_termal_d, float *reflectance_swir2_d, int width, int height)
// {
//   unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

//   // Map 1D position to 2D grid
//   unsigned int row = idx / width;
//   unsigned int col = idx % width;

//   if (idx < width * height)
//   {
//     unsigned int pos = row * width + col;

//     if (reflectance_blue_d[pos] <= 0)
//       reflectance_blue_d[pos] = NAN;
//     if (reflectance_green_d[pos] <= 0)
//       reflectance_green_d[pos] = NAN;
//     if (reflectance_red_d[pos] <= 0)
//       reflectance_red_d[pos] = NAN;
//     if (reflectance_nir_d[pos] <= 0)
//       reflectance_nir_d[pos] = NAN;
//     if (reflectance_swir1_d[pos] <= 0)
//       reflectance_swir1_d[pos] = NAN;
//     if (reflectance_termal_d[pos] <= 0)
//       reflectance_termal_d[pos] = NAN;
//     if (reflectance_swir2_d[pos] <= 0)
//       reflectance_swir2_d[pos] = NAN;
//   }
// }

// // Tensor corrections
// __global__ void invalid_lai_kernel(float *savi_d, float *lai_d, int width, int height)
// {
//   unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

//   // Map 1D position to 2D grid
//   unsigned int row = idx / width;
//   unsigned int col = idx % width;

//   if (idx < width * height)
//   {
//     unsigned int pos = row * width + col;

//     if (!isnan(savi_d[pos]) && savi_d[pos] < 0.1)
//       lai_d[pos] = 0;

//     if (lai_d[pos] < 0)
//       lai_d[pos] = 0;
//   }
// }

// __global__ void psi_kernel(float *d0, float *zom, float *kb1, float *L, float *ustar, float *rah, int height, int width)
// {
//   unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

//   // Map 1D position to 2D grid
//   unsigned int row = idx / width;
//   unsigned int col = idx % width;

//   if (idx < width * height)
//   {
//     unsigned int pos = row * width + col;

//     float y2 = pow((1 - (16 * (10 - d0[pos])) / L[pos]), 0.25);
//     float x200 = pow((1 - (16 * (10 - d0[pos])) / L[pos]), 0.25);

//     float psi2, psi200;
//     if (!isnan(L[pos]) && L[pos] > 0)
//     {
//       psi2 = -5 * ((10 - d0[pos]) / L[pos]);
//       psi200 = -5 * ((10 - d0[pos]) / L[pos]);
//     }
//     else
//     {
//       psi2 = 2 * log((1 + y2 * y2) / 2);
//       psi200 = 2 * log((1 + x200) / 2) + log((1 + x200 * x200) / 2) - 2 * atan(x200) + 0.5 * M_PI;
//     }

//     float ust = (VON_KARMAN * ustar[pos]) / (log((10 - d0[pos]) / zom[pos]) - psi200);

//     float zoh_terra = zom[pos] / pow(exp(1.0), (kb1[pos]));
//     float temp_rah1_corr_terra = (ust * VON_KARMAN);
//     float temp_rah2_corr_terra = log((10 - d0[pos]) / zom[pos]) - psi2;
//     float temp_rah3_corr_terra = temp_rah1_corr_terra * log(zom[pos] / zoh_terra);
//     float rah_final = (temp_rah1_corr_terra * temp_rah2_corr_terra) + temp_rah3_corr_terra;

//     ustar[pos] = ust;
//     rah[pos] = rah_final;
//   }
// }
