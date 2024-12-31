#include "products.h"
#include "kernels.cuh"

Products::Products() {}

Products::Products(uint32_t width_band, uint32_t height_band, int threads_num)
{
  this->threads_num = threads_num;
  this->blocks_num = (width_band * height_band + this->threads_num - 1) / this->threads_num;

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

  this->stop_condition = (int *)malloc(sizeof(int));  
  HANDLE_ERROR(cudaMalloc((void **)&this->stop_condition_d, sizeof(int)));

  HANDLE_ERROR(cudaMalloc((void **)&this->band_blue_d, nBytes_band));
  HANDLE_ERROR(cudaMalloc((void **)&this->band_green_d, nBytes_band));
  HANDLE_ERROR(cudaMalloc((void **)&this->band_red_d, nBytes_band));
  HANDLE_ERROR(cudaMalloc((void **)&this->band_nir_d, nBytes_band));
  HANDLE_ERROR(cudaMalloc((void **)&this->band_swir1_d, nBytes_band));
  HANDLE_ERROR(cudaMalloc((void **)&this->band_termal_d, nBytes_band));
  HANDLE_ERROR(cudaMalloc((void **)&this->band_swir2_d, nBytes_band));
  HANDLE_ERROR(cudaMalloc((void **)&this->tal_d, nBytes_band));

  HANDLE_ERROR(cudaMalloc((void **)&this->radiance_blue_d, nBytes_band));
  HANDLE_ERROR(cudaMalloc((void **)&this->radiance_green_d, nBytes_band));
  HANDLE_ERROR(cudaMalloc((void **)&this->radiance_red_d, nBytes_band));
  HANDLE_ERROR(cudaMalloc((void **)&this->radiance_nir_d, nBytes_band));
  HANDLE_ERROR(cudaMalloc((void **)&this->radiance_swir1_d, nBytes_band));
  HANDLE_ERROR(cudaMalloc((void **)&this->radiance_termal_d, nBytes_band));
  HANDLE_ERROR(cudaMalloc((void **)&this->radiance_swir2_d, nBytes_band));

  HANDLE_ERROR(cudaMalloc((void **)&this->reflectance_blue_d, nBytes_band));
  HANDLE_ERROR(cudaMalloc((void **)&this->reflectance_green_d, nBytes_band));
  HANDLE_ERROR(cudaMalloc((void **)&this->reflectance_red_d, nBytes_band));
  HANDLE_ERROR(cudaMalloc((void **)&this->reflectance_nir_d, nBytes_band));
  HANDLE_ERROR(cudaMalloc((void **)&this->reflectance_swir1_d, nBytes_band));
  HANDLE_ERROR(cudaMalloc((void **)&this->reflectance_termal_d, nBytes_band));
  HANDLE_ERROR(cudaMalloc((void **)&this->reflectance_swir2_d, nBytes_band));

  HANDLE_ERROR(cudaMalloc((void **)&this->albedo_d, nBytes_band));
  HANDLE_ERROR(cudaMalloc((void **)&this->ndvi_d, nBytes_band));
  HANDLE_ERROR(cudaMalloc((void **)&this->pai_d, nBytes_band));
  HANDLE_ERROR(cudaMalloc((void **)&this->lai_d, nBytes_band));
  HANDLE_ERROR(cudaMalloc((void **)&this->evi_d, nBytes_band));

  HANDLE_ERROR(cudaMalloc((void **)&this->enb_d, nBytes_band));
  HANDLE_ERROR(cudaMalloc((void **)&this->eo_d, nBytes_band));
  HANDLE_ERROR(cudaMalloc((void **)&this->ea_d, nBytes_band));
  HANDLE_ERROR(cudaMalloc((void **)&this->short_wave_radiation_d, nBytes_band));
  HANDLE_ERROR(cudaMalloc((void **)&this->large_wave_radiation_surface_d, nBytes_band));
  HANDLE_ERROR(cudaMalloc((void **)&this->large_wave_radiation_atmosphere_d, nBytes_band));
  HANDLE_ERROR(cudaMalloc((void **)&this->surface_temperature_d, nBytes_band));
  HANDLE_ERROR(cudaMalloc((void **)&this->net_radiation_d, nBytes_band));
  HANDLE_ERROR(cudaMalloc((void **)&this->soil_heat_d, nBytes_band));

  HANDLE_ERROR(cudaMalloc((void **)&this->zom_d, nBytes_band));
  HANDLE_ERROR(cudaMalloc((void **)&this->d0_d, nBytes_band));
  HANDLE_ERROR(cudaMalloc((void **)&this->kb1_d, nBytes_band));
  HANDLE_ERROR(cudaMalloc((void **)&this->ustar_d, nBytes_band));
  HANDLE_ERROR(cudaMalloc((void **)&this->rah_d, nBytes_band));
  HANDLE_ERROR(cudaMalloc((void **)&this->sensible_heat_flux_d, nBytes_band));

  HANDLE_ERROR(cudaMalloc((void **)&this->latent_heat_flux_d, nBytes_band));
  HANDLE_ERROR(cudaMalloc((void **)&this->net_radiation_24h_d, nBytes_band));
  HANDLE_ERROR(cudaMalloc((void **)&this->evapotranspiration_fraction_d, nBytes_band));
  HANDLE_ERROR(cudaMalloc((void **)&this->sensible_heat_flux_24h_d, nBytes_band));
  HANDLE_ERROR(cudaMalloc((void **)&this->latent_heat_flux_24h_d, nBytes_band));
  HANDLE_ERROR(cudaMalloc((void **)&this->evapotranspiration_24h_d, nBytes_band));
  HANDLE_ERROR(cudaMalloc((void **)&this->evapotranspiration_d, nBytes_band));
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

  HANDLE_ERROR(cudaFree(this->band_blue_d));
  HANDLE_ERROR(cudaFree(this->band_green_d));
  HANDLE_ERROR(cudaFree(this->band_red_d));
  HANDLE_ERROR(cudaFree(this->band_nir_d));
  HANDLE_ERROR(cudaFree(this->band_swir1_d));
  HANDLE_ERROR(cudaFree(this->band_termal_d));
  HANDLE_ERROR(cudaFree(this->band_swir2_d));
  HANDLE_ERROR(cudaFree(this->tal_d));

  HANDLE_ERROR(cudaFree(this->radiance_blue_d));
  HANDLE_ERROR(cudaFree(this->radiance_green_d));
  HANDLE_ERROR(cudaFree(this->radiance_red_d));
  HANDLE_ERROR(cudaFree(this->radiance_nir_d));
  HANDLE_ERROR(cudaFree(this->radiance_swir1_d));
  HANDLE_ERROR(cudaFree(this->radiance_termal_d));
  HANDLE_ERROR(cudaFree(this->radiance_swir2_d));

  HANDLE_ERROR(cudaFree(this->reflectance_blue_d));
  HANDLE_ERROR(cudaFree(this->reflectance_green_d));
  HANDLE_ERROR(cudaFree(this->reflectance_red_d));
  HANDLE_ERROR(cudaFree(this->reflectance_nir_d));
  HANDLE_ERROR(cudaFree(this->reflectance_swir1_d));
  HANDLE_ERROR(cudaFree(this->reflectance_termal_d));
  HANDLE_ERROR(cudaFree(this->reflectance_swir2_d));

  HANDLE_ERROR(cudaFree(this->albedo_d));
  HANDLE_ERROR(cudaFree(this->ndvi_d));
  HANDLE_ERROR(cudaFree(this->pai_d));
  HANDLE_ERROR(cudaFree(this->lai_d));
  HANDLE_ERROR(cudaFree(this->evi_d));

  HANDLE_ERROR(cudaFree(this->enb_d));
  HANDLE_ERROR(cudaFree(this->eo_d));
  HANDLE_ERROR(cudaFree(this->ea_d));
  HANDLE_ERROR(cudaFree(this->short_wave_radiation_d));
  HANDLE_ERROR(cudaFree(this->large_wave_radiation_surface_d));
  HANDLE_ERROR(cudaFree(this->large_wave_radiation_atmosphere_d));
  HANDLE_ERROR(cudaFree(this->surface_temperature_d));
  HANDLE_ERROR(cudaFree(this->net_radiation_d));
  HANDLE_ERROR(cudaFree(this->soil_heat_d));

  HANDLE_ERROR(cudaFree(this->zom_d));
  HANDLE_ERROR(cudaFree(this->d0_d));
  HANDLE_ERROR(cudaFree(this->kb1_d));
  HANDLE_ERROR(cudaFree(this->ustar_d));
  HANDLE_ERROR(cudaFree(this->rah_d));
  HANDLE_ERROR(cudaFree(this->sensible_heat_flux_d));

  HANDLE_ERROR(cudaFree(this->latent_heat_flux_d));
  HANDLE_ERROR(cudaFree(this->net_radiation_24h_d));
  HANDLE_ERROR(cudaFree(this->evapotranspiration_fraction_d));
  HANDLE_ERROR(cudaFree(this->sensible_heat_flux_24h_d));
  HANDLE_ERROR(cudaFree(this->latent_heat_flux_24h_d));
  HANDLE_ERROR(cudaFree(this->evapotranspiration_24h_d));
  HANDLE_ERROR(cudaFree(this->evapotranspiration_d));
};

string Products::radiance_function(MTL mtl)
{
  int64_t initial_time, final_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  cudaEventRecord(start, 0);
  rad_kernel<<<this->blocks_num, this->threads_num>>>(band_blue_d, radiance_blue_d, mtl.rad_add_d, mtl.rad_mult_d, PARAM_BAND_BLUE_INDEX, width_band, height_band);
  rad_kernel<<<this->blocks_num, this->threads_num>>>(band_green_d, radiance_green_d, mtl.rad_add_d, mtl.rad_mult_d, PARAM_BAND_GREEN_INDEX, width_band, height_band);
  rad_kernel<<<this->blocks_num, this->threads_num>>>(band_red_d, radiance_red_d, mtl.rad_add_d, mtl.rad_mult_d, PARAM_BAND_RED_INDEX, width_band, height_band);
  rad_kernel<<<this->blocks_num, this->threads_num>>>(band_nir_d, radiance_nir_d, mtl.rad_add_d, mtl.rad_mult_d, PARAM_BAND_NIR_INDEX, width_band, height_band);
  rad_kernel<<<this->blocks_num, this->threads_num>>>(band_swir1_d, radiance_swir1_d, mtl.rad_add_d, mtl.rad_mult_d, PARAM_BAND_SWIR1_INDEX, width_band, height_band);
  rad_kernel<<<this->blocks_num, this->threads_num>>>(band_termal_d, radiance_termal_d, mtl.rad_add_d, mtl.rad_mult_d, PARAM_BAND_TERMAL_INDEX, width_band, height_band);
  rad_kernel<<<this->blocks_num, this->threads_num>>>(band_swir2_d, radiance_swir2_d, mtl.rad_add_d, mtl.rad_mult_d, PARAM_BAND_SWIR2_INDEX, width_band, height_band);
  cudaEventRecord(stop, 0);

  float cuda_time = 0;
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&cuda_time, start, stop);
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  return "KERNELS,RADIANCE," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
}

string Products::reflectance_function(MTL mtl)
{
  const float sin_sun = sin(mtl.sun_elevation * PI / 180);

  int64_t initial_time, final_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  cudaEventRecord(start);
  ref_kernel<<<this->blocks_num, this->threads_num>>>(band_blue_d, reflectance_blue_d, mtl.ref_add_d, mtl.ref_mult_d, sin_sun, PARAM_BAND_BLUE_INDEX, width_band, height_band);
  ref_kernel<<<this->blocks_num, this->threads_num>>>(band_green_d, reflectance_green_d, mtl.ref_add_d, mtl.ref_mult_d, sin_sun, PARAM_BAND_GREEN_INDEX, width_band, height_band);
  ref_kernel<<<this->blocks_num, this->threads_num>>>(band_red_d, reflectance_red_d, mtl.ref_add_d, mtl.ref_mult_d, sin_sun, PARAM_BAND_RED_INDEX, width_band, height_band);
  ref_kernel<<<this->blocks_num, this->threads_num>>>(band_nir_d, reflectance_nir_d, mtl.ref_add_d, mtl.ref_mult_d, sin_sun, PARAM_BAND_NIR_INDEX, width_band, height_band);
  ref_kernel<<<this->blocks_num, this->threads_num>>>(band_swir1_d, reflectance_swir1_d, mtl.ref_add_d, mtl.ref_mult_d, sin_sun, PARAM_BAND_SWIR1_INDEX, width_band, height_band);
  ref_kernel<<<this->blocks_num, this->threads_num>>>(band_termal_d, reflectance_termal_d, mtl.ref_add_d, mtl.ref_mult_d, sin_sun, PARAM_BAND_TERMAL_INDEX, width_band, height_band);
  ref_kernel<<<this->blocks_num, this->threads_num>>>(band_swir2_d, reflectance_swir2_d, mtl.ref_add_d, mtl.ref_mult_d, sin_sun, PARAM_BAND_SWIR2_INDEX, width_band, height_band);
  cudaEventRecord(stop);

  float cuda_time = 0;
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&cuda_time, start, stop);
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  return "KERNELS,REFLECTANCE," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
}

string Products::albedo_function(MTL mtl)
{
  int64_t initial_time, final_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  cudaEventRecord(start);
  albedo_kernel<<<this->blocks_num, this->threads_num>>>(reflectance_blue_d, reflectance_green_d, reflectance_red_d, reflectance_nir_d, reflectance_swir1_d, reflectance_swir2_d,
                                                         tal_d, albedo_d, mtl.ref_w_coeff_d, width_band, height_band);
  cudaEventRecord(stop);

  float cuda_time = 0;
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&cuda_time, start, stop);
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  return "KERNELS,ALBEDO," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
}

string Products::ndvi_function()
{
  int64_t initial_time, final_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  cudaEventRecord(start);
  ndvi_kernel<<<this->blocks_num, this->threads_num>>>(reflectance_nir_d, reflectance_red_d, ndvi_d, width_band, height_band);
  cudaEventRecord(stop);

  float cuda_time = 0;
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&cuda_time, start, stop);
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  return "KERNELS,NDVI," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::pai_function()
{
  int64_t initial_time, final_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  cudaEventRecord(start);
  pai_kernel<<<this->blocks_num, this->threads_num>>>(reflectance_nir_d, reflectance_red_d, pai_d, width_band, height_band);
  cudaEventRecord(stop);

  float cuda_time = 0;
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&cuda_time, start, stop);
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  return "KERNELS,PAI," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::lai_function()
{
  int64_t initial_time, final_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  cudaEventRecord(start);
  lai_kernel<<<this->blocks_num, this->threads_num>>>(reflectance_nir_d, reflectance_red_d, lai_d, width_band, height_band);
  cudaEventRecord(stop);

  float cuda_time = 0;
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&cuda_time, start, stop);
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  return "KERNELS,LAI," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::evi_function()
{
  int64_t initial_time, final_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  cudaEventRecord(start);
  evi_kernel<<<this->blocks_num, this->threads_num>>>(reflectance_nir_d, reflectance_red_d, reflectance_blue_d, evi_d, width_band, height_band);
  cudaEventRecord(stop);

  float cuda_time = 0;
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&cuda_time, start, stop);
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  return "KERNELS,EVI," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::enb_emissivity_function()
{
  int64_t initial_time, final_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  cudaEventRecord(start);
  enb_kernel<<<this->blocks_num, this->threads_num>>>(lai_d, ndvi_d, enb_d, width_band, height_band);
  cudaEventRecord(stop);

  float cuda_time = 0;
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&cuda_time, start, stop);
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  return "KERNELS,ENB_EMISSIVITY," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::eo_emissivity_function()
{
  int64_t initial_time, final_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  cudaEventRecord(start);
  eo_kernel<<<this->blocks_num, this->threads_num>>>(lai_d, ndvi_d, eo_d, width_band, height_band);
  cudaEventRecord(stop);

  float cuda_time = 0;
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&cuda_time, start, stop);
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  return "KERNELS,EO_EMISSIVITY," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::ea_emissivity_function()
{
  int64_t initial_time, final_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  cudaEventRecord(start);
  ea_kernel<<<this->blocks_num, this->threads_num>>>(tal_d, ea_d, width_band, height_band);
  cudaEventRecord(stop);

  float cuda_time = 0;
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&cuda_time, start, stop);
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  return "KERNELS,EA_EMISSIVITY," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
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

  int64_t initial_time, final_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  cudaEventRecord(start);
  surface_temperature_kernel<<<this->blocks_num, this->threads_num>>>(enb_d, radiance_termal_d, surface_temperature_d, k1, k2, width_band, height_band);
  cudaEventRecord(stop);

  float cuda_time = 0;
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&cuda_time, start, stop);
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  return "KERNELS,SURFACE_TEMPERATURE," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::short_wave_radiation_function(MTL mtl)
{
  int64_t initial_time, final_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  cudaEventRecord(start);
  short_wave_radiation_kernel<<<this->blocks_num, this->threads_num>>>(tal_d, short_wave_radiation_d, mtl.sun_elevation, mtl.distance_earth_sun, PI, width_band, height_band);
  cudaEventRecord(stop);

  float cuda_time = 0;
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&cuda_time, start, stop);
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  return "KERNELS,SHORT_WAVE_RADIATION," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::large_wave_radiation_surface_function()
{
  int64_t initial_time, final_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  cudaEventRecord(start);
  large_wave_radiation_surface_kernel<<<this->blocks_num, this->threads_num>>>(surface_temperature_d, eo_d, large_wave_radiation_surface_d, width_band, height_band);
  cudaEventRecord(stop);

  float cuda_time = 0;
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&cuda_time, start, stop);
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  return "KERNELS,LARGE_WAVE_RADIATION_SURFACE," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::large_wave_radiation_atmosphere_function(float temperature)
{
  int64_t initial_time, final_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  cudaEventRecord(start);
  large_wave_radiation_atmosphere_kernel<<<this->blocks_num, this->threads_num>>>(ea_d, large_wave_radiation_atmosphere_d, temperature, width_band, height_band);
  cudaEventRecord(stop);

  float cuda_time = 0;
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&cuda_time, start, stop);
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  return "KERNELS,LARGE_WAVE_RADIATION_ATMOSPHERE," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::net_radiation_function()
{
  int64_t initial_time, final_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  cudaEventRecord(start);
  net_radiation_kernel<<<this->blocks_num, this->threads_num>>>(short_wave_radiation_d, albedo_d, large_wave_radiation_atmosphere_d, large_wave_radiation_surface_d, eo_d, net_radiation_d, width_band, height_band);
  cudaEventRecord(stop);

  float cuda_time = 0;
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&cuda_time, start, stop);
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  return "KERNELS,NET_RADIATION," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::soil_heat_flux_function()
{
  int64_t initial_time, final_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  cudaEventRecord(start);
  soil_heat_kernel<<<this->blocks_num, this->threads_num>>>(ndvi_d, albedo_d, surface_temperature_d, net_radiation_d, soil_heat_d, width_band, height_band);
  cudaEventRecord(stop);

  float cuda_time = 0;
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&cuda_time, start, stop);
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  return "KERNELS,SOIL_HEAT_FLUX," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::d0_fuction()
{
  float CD1 = 20.6;
  float HGHT = 4;

  int64_t initial_time, final_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  cudaEventRecord(start);
  d0_kernel<<<this->blocks_num, this->threads_num>>>(pai_d, d0_d, CD1, HGHT, width_band, height_band);
  cudaEventRecord(stop);

  float cuda_time = 0;
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&cuda_time, start, stop);
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  return "KERNELS,D0," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::zom_fuction(float A_ZOM, float B_ZOM)
{
  int64_t initial_time, final_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  cudaEventRecord(start);
  zom_kernel<<<this->blocks_num, this->threads_num>>>(d0_d, pai_d, zom_d, A_ZOM, B_ZOM, width_band, height_band);
  cudaEventRecord(stop);

  float cuda_time = 0;
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&cuda_time, start, stop);
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  return "KERNELS,ZOM," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::ustar_fuction(float u10)
{
  int64_t initial_time, final_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  cudaEventRecord(start);
  ustar_kernel<<<this->blocks_num, this->threads_num>>>(zom_d, d0_d, ustar_d, u10, width_band, height_band);
  cudaEventRecord(stop);

  float cuda_time = 0;
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&cuda_time, start, stop);
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  return "KERNELS,USTAR," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::kb_function(float ndvi_max, float ndvi_min)
{
  int64_t initial_time, final_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  cudaEventRecord(start);
  kb_kernel<<<this->blocks_num, this->threads_num>>>(zom_d, ustar_d, pai_d, kb1_d, ndvi_d, width_band, height_band, ndvi_max, ndvi_min);
  cudaEventRecord(stop);

  float cuda_time = 0;
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&cuda_time, start, stop);
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  return "KERNELS,KB1," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::aerodynamic_resistance_fuction()
{
  int64_t initial_time, final_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  cudaEventRecord(start);
  aerodynamic_resistance_kernel<<<this->blocks_num, this->threads_num>>>(zom_d, d0_d, ustar_d, kb1_d, rah_d, width_band, height_band);
  cudaEventRecord(stop);

  float cuda_time = 0;
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&cuda_time, start, stop);
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  return "KERNELS,RAH_INI," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::sensible_heat_flux_function(Candidate *d_hotCandidates, Candidate *d_coldCandidates)
{
  int64_t initial_time, final_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  cudaEventRecord(start);
  sensible_heat_flux_kernel<<<this->blocks_num, this->threads_num>>>(d_hotCandidates, d_coldCandidates, surface_temperature_d, 
                                                                     rah_d, net_radiation_d, soil_heat_d, sensible_heat_flux_d, width_band, height_band);

  cudaEventRecord(stop);

  float cuda_time = 0;
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&cuda_time, start, stop);
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  return "KERNELS,SENSIBLE_HEAT_FLUX," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::latent_heat_flux_function()
{
  int64_t initial_time, final_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  cudaEventRecord(start);
  latent_heat_flux_kernel<<<this->blocks_num, this->threads_num>>>(net_radiation_d, soil_heat_d, sensible_heat_flux_d, latent_heat_flux_d, width_band, height_band);
  cudaEventRecord(stop);

  float cuda_time = 0;
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&cuda_time, start, stop);
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  return "KERNELS,LATENT_HEAT_FLUX," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::net_radiation_24h_function(float Ra24h, float Rs24h)
{
  int64_t initial_time, final_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  cudaEventRecord(start);
  net_radiation_24h_kernel<<<this->blocks_num, this->threads_num>>>(albedo_d, Rs24h, Ra24h, net_radiation_24h_d, width_band, height_band);
  cudaEventRecord(stop);

  float cuda_time = 0;
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&cuda_time, start, stop);
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  return "KERNELS,NET_RADIATION_24H," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::evapotranspiration_fraction_fuction()
{
  int64_t initial_time, final_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  cudaEventRecord(start);
  evapotranspiration_fraction_kernel<<<this->blocks_num, this->threads_num>>>(net_radiation_d, soil_heat_d, latent_heat_flux_d, evapotranspiration_fraction_d, width_band, height_band);
  cudaEventRecord(stop);

  float cuda_time = 0;
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&cuda_time, start, stop);
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  return "KERNELS,EVAPOTRANSPIRATION_FRACTION," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::sensible_heat_flux_24h_fuction()
{
  int64_t initial_time, final_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  cudaEventRecord(start);
  sensible_heat_flux_24h_kernel<<<this->blocks_num, this->threads_num>>>(net_radiation_24h_d, evapotranspiration_fraction_d, sensible_heat_flux_24h_d, width_band, height_band);
  cudaEventRecord(stop);

  float cuda_time = 0;
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&cuda_time, start, stop);
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  return "KERNELS,SENSIBLE_HEAT_FLUX_24H," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::latent_heat_flux_24h_function()
{
  int64_t initial_time, final_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  cudaEventRecord(start);
  latent_heat_flux_24h_kernel<<<this->blocks_num, this->threads_num>>>(net_radiation_24h_d, evapotranspiration_fraction_d, latent_heat_flux_24h_d, width_band, height_band);
  cudaEventRecord(stop);

  float cuda_time = 0;
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&cuda_time, start, stop);
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  return "KERNELS,LATENT_HEAT_FLUX_24H," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::evapotranspiration_24h_function(Station station)
{
  int64_t initial_time, final_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  cudaEventRecord(start);
  evapotranspiration_24h_kernel<<<this->blocks_num, this->threads_num>>>(latent_heat_flux_24h_d, evapotranspiration_24h_d, station.v7_max, station.v7_min, width_band, height_band);
  cudaEventRecord(stop);

  float cuda_time = 0;
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&cuda_time, start, stop);
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  return "KERNELS,EVAPOTRANSPIRATION_24H," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::evapotranspiration_function()
{
  int64_t initial_time, final_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  cudaEventRecord(start);
  evapotranspiration_kernel<<<this->blocks_num, this->threads_num>>>(net_radiation_24h_d, evapotranspiration_fraction_d, evapotranspiration_d, width_band, height_band);
  cudaEventRecord(stop);

  float cuda_time = 0;
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&cuda_time, start, stop);
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  return "KERNELS,EVAPOTRANSPIRATION," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::rah_correction_function_blocks_STEEP(Candidate *d_hotCandidates, Candidate *d_coldCandidates,
                                                      float ndvi_min, float ndvi_max)
{
  string result = "";
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  int64_t initial_time, final_time;

  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  // ========= CUDA Setup
  int dev = 0;
  cudaDeviceProp deviceProp;
  HANDLE_ERROR(cudaGetDeviceProperties(&deviceProp, dev));
  HANDLE_ERROR(cudaSetDevice(dev));

  cudaEventRecord(start);
  for (int i = 0; i < 2; i++)
  {
    rah_correction_cycle_STEEP<<<this->blocks_num, this->threads_num>>>(d_hotCandidates, d_coldCandidates, ndvi_d,
                                                                        surface_temperature_d, d0_d, kb1_d, zom_d, ustar_d, rah_d, sensible_heat_flux_d,
                                                                        ndvi_max, ndvi_min, height_band, width_band);
  }
  cudaEventRecord(stop);

  float cuda_time = 0;
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&cuda_time, start, stop);
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
  return "KERNELS,RAH_CYCLE," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
}

string Products::rah_correction_function_blocks_ASEBAL(Candidate *d_hotCandidates, Candidate *d_coldCandidates,
                                                       float ndvi_min, float ndvi_max, float u200)
{
  string result = "";
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  int64_t initial_time, final_time;

  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  // ========= CUDA Setup
  int dev = 0;
  cudaDeviceProp deviceProp;
  HANDLE_ERROR(cudaGetDeviceProperties(&deviceProp, dev));
  HANDLE_ERROR(cudaSetDevice(dev));

  cudaEventRecord(start);
  int i = 0;
  while (true)
  {
    rah_correction_cycle_ASEBAL<<<this->blocks_num, this->threads_num>>>(d_hotCandidates, d_coldCandidates, 
                                                                         ndvi_d, surface_temperature_d, kb1_d, zom_d, ustar_d,
                                                                         rah_d, sensible_heat_flux_d, ndvi_max, ndvi_min,
                                                                         u200, height_band, width_band, stop_condition_d);

    HANDLE_ERROR(cudaMemcpy(stop_condition, stop_condition_d, sizeof(int), cudaMemcpyDeviceToHost));

    if (i > 0 && *stop_condition)
      break;
    else
      i++;
  }

  cudaEventRecord(stop);

  float cuda_time = 0;
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&cuda_time, start, stop);
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
  return "KERNELS,RAH_CYCLE," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
}
