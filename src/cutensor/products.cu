#include "products.h"
#include "kernels.cuh"

Products::Products() {}

Products::Products(uint32_t width_band, uint32_t height_band, int threads_num)
{
  this->tensors = Tensor(width_band, height_band);

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
  HANDLE_ERROR(cudaMalloc((void **)&this->savi_d, nBytes_band));
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

  // === Used in tensor implementation ===
  this->only1 = (float *)malloc(nBytes_band);
  HANDLE_ERROR(cudaMalloc((void **)&this->only1_d, nBytes_band));
  HANDLE_ERROR(cudaMalloc((void **)&this->tensor_aux1_d, nBytes_band));
  HANDLE_ERROR(cudaMalloc((void **)&this->tensor_aux2_d, nBytes_band));
  HANDLE_ERROR(cudaMalloc((void **)&this->Re_star_d, nBytes_band));
  HANDLE_ERROR(cudaMalloc((void **)&this->Ct_star_d, nBytes_band));
  HANDLE_ERROR(cudaMalloc((void **)&this->beta_d, nBytes_band));
  HANDLE_ERROR(cudaMalloc((void **)&this->nec_terra_d, nBytes_band));
  HANDLE_ERROR(cudaMalloc((void **)&this->kb1_fst_part_d, nBytes_band));
  HANDLE_ERROR(cudaMalloc((void **)&this->kb1_sec_part_d, nBytes_band));
  HANDLE_ERROR(cudaMalloc((void **)&this->kb1s_d, nBytes_band));
  HANDLE_ERROR(cudaMalloc((void **)&this->fc_d, nBytes_band));
  HANDLE_ERROR(cudaMalloc((void **)&this->fs_d, nBytes_band));
  HANDLE_ERROR(cudaMalloc((void **)&this->fspow_d, nBytes_band));
  HANDLE_ERROR(cudaMalloc((void **)&this->fcpow_d, nBytes_band));
  // === Used in tensor implementation ===
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
  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add, (void *)&mtl.rad_mult[PARAM_BAND_BLUE_INDEX], band_blue_d, (void *)&mtl.rad_add[PARAM_BAND_BLUE_INDEX], only1_d, radiance_blue_d, tensors.stream));
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add, (void *)&mtl.rad_mult[PARAM_BAND_GREEN_INDEX], band_green_d, (void *)&mtl.rad_add[PARAM_BAND_GREEN_INDEX], only1_d, radiance_green_d, tensors.stream));
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add, (void *)&mtl.rad_mult[PARAM_BAND_RED_INDEX], band_red_d, (void *)&mtl.rad_add[PARAM_BAND_RED_INDEX], only1_d, radiance_red_d, tensors.stream));
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add, (void *)&mtl.rad_mult[PARAM_BAND_NIR_INDEX], band_nir_d, (void *)&mtl.rad_add[PARAM_BAND_NIR_INDEX], only1_d, radiance_nir_d, tensors.stream));
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add, (void *)&mtl.rad_mult[PARAM_BAND_SWIR1_INDEX], band_swir1_d, (void *)&mtl.rad_add[PARAM_BAND_SWIR1_INDEX], only1_d, radiance_swir1_d, tensors.stream));
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add, (void *)&mtl.rad_mult[PARAM_BAND_TERMAL_INDEX], band_termal_d, (void *)&mtl.rad_add[PARAM_BAND_TERMAL_INDEX], only1_d, radiance_termal_d, tensors.stream));
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add, (void *)&mtl.rad_mult[PARAM_BAND_SWIR2_INDEX], band_swir2_d, (void *)&mtl.rad_add[PARAM_BAND_SWIR2_INDEX], only1_d, radiance_swir2_d, tensors.stream));

  // invalid_rad_kernel<<<this->blocks_num, this->threads_num>>>(radiance_blue_d, radiance_green_d, radiance_red_d, radiance_nir_d, radiance_swir1_d, radiance_termal_d, radiance_swir2_d, this->width_band, this->height_band);
  // rad_kernel<<<this->blocks_num, this->threads_num>>>(band_blue_d, band_green_d, band_red_d, band_nir_d, band_swir1_d, band_termal_d, band_swir2_d,
  //                                                     radiance_blue_d, radiance_green_d, radiance_red_d, radiance_nir_d, radiance_swir1_d, radiance_termal_d, radiance_swir2_d,
  //                                                     mtl.rad_add_d, mtl.rad_mult_d, width_band, height_band);

  // HANDLE_ERROR(cudaDeviceSynchronize());
  // HANDLE_ERROR(cudaGetLastError());

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  HANDLE_ERROR(cudaMemcpy(radiance_blue, radiance_blue_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(radiance_green, radiance_green_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(radiance_red, radiance_red_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(radiance_nir, radiance_nir_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(radiance_swir1, radiance_swir1_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(radiance_termal, radiance_termal_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(radiance_swir2, radiance_swir2_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));

  return "CUTENSOR,RADIANCE," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
}

string Products::reflectance_function(MTL mtl)
{
  const float sin_sun = 1 / sin(mtl.sun_elevation * PI / 180);

  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  HANDLE_CUTENSOR_ERROR(cutensorElementwiseTrinaryExecute(tensors.handle, tensors.tensor_plan_trinity_add_mult, (void *)&mtl.ref_mult[PARAM_BAND_BLUE_INDEX], band_blue_d, (void *)&mtl.ref_add[PARAM_BAND_BLUE_INDEX], only1_d, (void *)&sin_sun, only1_d, reflectance_blue_d, tensors.stream));
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseTrinaryExecute(tensors.handle, tensors.tensor_plan_trinity_add_mult, (void *)&mtl.ref_mult[PARAM_BAND_GREEN_INDEX], band_green_d, (void *)&mtl.ref_add[PARAM_BAND_GREEN_INDEX], only1_d, (void *)&sin_sun, only1_d, reflectance_green_d, tensors.stream));
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseTrinaryExecute(tensors.handle, tensors.tensor_plan_trinity_add_mult, (void *)&mtl.ref_mult[PARAM_BAND_RED_INDEX], band_red_d, (void *)&mtl.ref_add[PARAM_BAND_RED_INDEX], only1_d, (void *)&sin_sun, only1_d, reflectance_red_d, tensors.stream));
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseTrinaryExecute(tensors.handle, tensors.tensor_plan_trinity_add_mult, (void *)&mtl.ref_mult[PARAM_BAND_NIR_INDEX], band_nir_d, (void *)&mtl.ref_add[PARAM_BAND_NIR_INDEX], only1_d, (void *)&sin_sun, only1_d, reflectance_nir_d, tensors.stream));
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseTrinaryExecute(tensors.handle, tensors.tensor_plan_trinity_add_mult, (void *)&mtl.ref_mult[PARAM_BAND_SWIR1_INDEX], band_swir1_d, (void *)&mtl.ref_add[PARAM_BAND_SWIR1_INDEX], only1_d, (void *)&sin_sun, only1_d, reflectance_swir1_d, tensors.stream));
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseTrinaryExecute(tensors.handle, tensors.tensor_plan_trinity_add_mult, (void *)&mtl.ref_mult[PARAM_BAND_TERMAL_INDEX], band_termal_d, (void *)&mtl.ref_add[PARAM_BAND_TERMAL_INDEX], only1_d, (void *)&sin_sun, only1_d, reflectance_termal_d, tensors.stream));
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseTrinaryExecute(tensors.handle, tensors.tensor_plan_trinity_add_mult, (void *)&mtl.ref_mult[PARAM_BAND_SWIR2_INDEX], band_swir2_d, (void *)&mtl.ref_add[PARAM_BAND_SWIR2_INDEX], only1_d, (void *)&sin_sun, only1_d, reflectance_swir2_d, tensors.stream));

  // invalid_ref_kernel<<<this->blocks_num, this->threads_num>>>(reflectance_blue_d, reflectance_green_d, reflectance_red_d, reflectance_nir_d, reflectance_swir1_d, reflectance_termal_d, reflectance_swir2_d, this->width_band, this->height_band);
  // ref_kernel<<<this->blocks_num, this->threads_num>>>(band_blue_d, band_green_d, band_red_d, band_nir_d, band_swir1_d, band_termal_d, band_swir2_d,
  //                                                   reflectance_blue_d, reflectance_green_d, reflectance_red_d, reflectance_nir_d, reflectance_swir1_d, reflectance_termal_d, reflectance_swir2_d,
  //                                                   mtl.ref_add_d, mtl.ref_mult_d, sin_sun, width_band, height_band);

  // HANDLE_ERROR(cudaDeviceSynchronize());
  // HANDLE_ERROR(cudaGetLastError());

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  HANDLE_ERROR(cudaMemcpy(reflectance_blue, reflectance_blue_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(reflectance_green, reflectance_green_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(reflectance_red, reflectance_red_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(reflectance_nir, reflectance_nir_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(reflectance_swir1, reflectance_swir1_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(reflectance_termal, reflectance_termal_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(reflectance_swir2, reflectance_swir2_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));

  return "CUTENSOR,REFLECTANCE," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
}

string Products::albedo_function(MTL mtl)
{
  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  float pos1 = 1;
  float neg003 = -0.03;
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_rcp, (void *)&pos1, tal_d, (void *)&pos1, tal_d, tensor_aux1_d, tensors.stream));
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add, (void *)&mtl.ref_w_coeff[PARAM_BAND_BLUE_INDEX], reflectance_blue_d, (void *)&mtl.ref_w_coeff[PARAM_BAND_GREEN_INDEX], reflectance_green_d, albedo_d, tensors.stream));
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add, (void *)&pos1, albedo_d, (void *)&mtl.ref_w_coeff[PARAM_BAND_RED_INDEX], reflectance_red_d, albedo_d, tensors.stream));
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add, (void *)&pos1, albedo_d, (void *)&mtl.ref_w_coeff[PARAM_BAND_NIR_INDEX], reflectance_nir_d, albedo_d, tensors.stream));
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add, (void *)&pos1, albedo_d, (void *)&mtl.ref_w_coeff[PARAM_BAND_SWIR1_INDEX], reflectance_swir1_d, albedo_d, tensors.stream));
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add, (void *)&pos1, albedo_d, (void *)&mtl.ref_w_coeff[PARAM_BAND_SWIR2_INDEX], reflectance_swir2_d, albedo_d, tensors.stream));
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseTrinaryExecute(tensors.handle, tensors.tensor_plan_trinity_add_mult, (void *)&pos1, albedo_d, (void *)&neg003, only1_d, (void *)&pos1, tensor_aux1_d, albedo_d, tensors.stream));

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  HANDLE_ERROR(cudaMemcpy(albedo, albedo_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));

  return "CUTENSOR,ALBEDO," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
}

string Products::invalid_rad_ref_function()
{
  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  invalid_rad_ref_kernel<<<this->blocks_num, this->threads_num>>>(albedo_d, radiance_blue_d, radiance_green_d, radiance_red_d, radiance_nir_d, radiance_swir1_d, radiance_termal_d, radiance_swir2_d,
                                                                  reflectance_blue_d, reflectance_green_d, reflectance_red_d, reflectance_nir_d, reflectance_swir1_d, reflectance_termal_d, reflectance_swir2_d,
                                                                  this->width_band, this->height_band);

  HANDLE_ERROR(cudaDeviceSynchronize());
  HANDLE_ERROR(cudaGetLastError());

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  HANDLE_ERROR(cudaMemcpy(radiance_blue, radiance_blue_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(radiance_green, radiance_green_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(radiance_red, radiance_red_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(radiance_nir, radiance_nir_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(radiance_swir1, radiance_swir1_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(radiance_termal, radiance_termal_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(radiance_swir2, radiance_swir2_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));

  HANDLE_ERROR(cudaMemcpy(reflectance_blue, reflectance_blue_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(reflectance_green, reflectance_green_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(reflectance_red, reflectance_red_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(reflectance_nir, reflectance_nir_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(reflectance_swir1, reflectance_swir1_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(reflectance_termal, reflectance_termal_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(reflectance_swir2, reflectance_swir2_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));

  HANDLE_ERROR(cudaMemcpy(albedo, albedo_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));

  return "CUDACORE,RAD_REF," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
}

string Products::ndvi_function()
{
  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  float pos1 = 1;
  float neg1 = -1;
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add, (void *)&pos1, reflectance_nir_d, (void *)&neg1, reflectance_red_d, tensor_aux1_d, tensors.stream));
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add, (void *)&pos1, reflectance_nir_d, (void *)&pos1, reflectance_red_d, tensor_aux2_d, tensors.stream));
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_div, (void *)&pos1, tensor_aux1_d, (void *)&pos1, tensor_aux2_d, ndvi_d, tensors.stream));

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  HANDLE_ERROR(cudaMemcpy(ndvi, ndvi_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));

  return "CUTENSOR,NDVI," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::pai_function()
{
  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  float pos0 = 0;
  float pos1 = 1;
  float pos31 = 3.1;
  float pos101 = 10.1;
  float neg1 = -1;
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_sqtr_add, (void *)&pos1, reflectance_nir_d, (void *)&neg1, reflectance_red_d, tensor_aux1_d, tensors.stream));
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add, (void *)&pos101, tensor_aux1_d, (void *)&pos31, only1_d, pai_d, tensors.stream));
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_max, (void *)&pos1, pai_d, (void *)&pos0, only1_d, pai_d, tensors.stream));

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  HANDLE_ERROR(cudaMemcpy(pai, pai_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));

  return "CUTENSOR,PAI," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::lai_function()
{
  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  // float neg1 = -1;
  // float pos0 = 0;
  // float pos1 = 1;
  // float pos6 = 6;
  // float pos05 = 0.5; // L
  // float pos15 = 1.5; // 1 + L
  // float pos069 = 0.69;
  // float frc059 = 1 / 0.59;
  // float frc091 = 1 / 0.91;

  // HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add, (void *)&pos1, reflectance_nir_d, (void *)&neg1, reflectance_red_d, tensor_aux1_d, tensors.stream));
  // HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add, (void *)&pos1, reflectance_nir_d, (void *)&pos1, reflectance_red_d, tensor_aux2_d, tensors.stream));
  // HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add, (void *)&pos1, tensor_aux2_d, (void *)&pos05, only1_d, tensor_aux2_d, tensors.stream));
  // HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_div, (void *)&pos15, tensor_aux1_d, (void *)&pos1, tensor_aux2_d, savi_d, tensors.stream));
  // HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_min, (void *)&pos1, savi_d, (void *)&pos6, only1_d, lai_d, tensors.stream));
  // HANDLE_CUTENSOR_ERROR(cutensorElementwiseTrinaryExecute(tensors.handle, tensors.tensor_plan_trinity_mult_add, (void *)&pos069, only1_d, (void *)&neg1, lai_d, (void *)&frc059, only1_d, tensor_aux1_d, tensors.stream));
  // HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_log_mul, (void *)&frc091, only1_d, (void *)&neg1, tensor_aux1_d, lai_d, tensors.stream));

  lai_kernel<<<this->blocks_num, this->threads_num>>>(reflectance_nir_d, reflectance_red_d, lai_d, width_band, height_band);

  HANDLE_ERROR(cudaDeviceSynchronize());
  HANDLE_ERROR(cudaGetLastError());

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  HANDLE_ERROR(cudaMemcpy(lai, lai_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));

  return "CUDACORE,LAI," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::evi_function()
{
  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  float pos0 = 0;
  float pos1 = 1;
  float pos25 = 2.5;
  float pos6 = 6;
  float neg1 = -1;
  float neg75 = -7.5;
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add, (void *)&pos1, reflectance_nir_d, (void *)&neg1, reflectance_red_d, tensor_aux1_d, tensors.stream));
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add, (void *)&pos6, reflectance_red_d, (void *)&neg75, reflectance_blue_d, tensor_aux2_d, tensors.stream));
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add, (void *)&pos1, reflectance_nir_d, (void *)&pos1, tensor_aux2_d, tensor_aux2_d, tensors.stream));
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add, (void *)&pos1, only1_d, (void *)&pos1, tensor_aux2_d, tensor_aux2_d, tensors.stream));
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_div, (void *)&pos25, tensor_aux1_d, (void *)&pos1, tensor_aux2_d, evi_d, tensors.stream));
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_max, (void *)&pos1, evi_d, (void *)&pos0, only1_d, evi_d, tensors.stream));

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  HANDLE_ERROR(cudaMemcpy(evi, evi_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));

  return "CUTENSOR,EVI," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::enb_emissivity_function()
{
  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  // float pos097 = 0.97;
  // float pos0033 = 0.0033;
  // HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add, (void *)&pos097, only1_d, (void *)&pos0033, lai_d, enb_d, tensors.stream));
  // TODO: treat the values -> lai = 0 and (ndvi < 0 and lai > 2.99)

  enb_kernel<<<this->blocks_num, this->threads_num>>>(lai_d, ndvi_d, enb_d, width_band, height_band);

  HANDLE_ERROR(cudaDeviceSynchronize());
  HANDLE_ERROR(cudaGetLastError());

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  HANDLE_ERROR(cudaMemcpy(enb_emissivity, enb_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));

  return "CUDACORE,ENB_EMISSIVITY," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::eo_emissivity_function()
{
  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  // float pos095 = 0.95;
  // float pos001 = 0.01;
  // HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add, (void *)&pos095, only1_d, (void *)&pos001, lai_d, eo_d, tensors.stream));

  eo_kernel<<<this->blocks_num, this->threads_num>>>(lai_d, ndvi_d, eo_d, width_band, height_band);

  HANDLE_ERROR(cudaDeviceSynchronize());
  HANDLE_ERROR(cudaGetLastError());

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  HANDLE_ERROR(cudaMemcpy(eo_emissivity, eo_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));

  return "CUDACORE,EO_EMISSIVITY," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::ea_emissivity_function()
{
  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  float pos1 = 1;
  float pos085 = 0.85;
  float pos009 = 0.09;
  float neg1 = -1;
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_log_mul, (void *)&pos1, only1_d, (void *)&neg1, tal_d, tensor_aux1_d, tensors.stream));
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_log_mul, (void *)&pos009, only1_d, (void *)&pos1, tensor_aux1_d, tensor_aux1_d, tensors.stream));
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_exp_mul, (void *)&pos085, only1_d, (void *)&pos1, tensor_aux1_d, ea_d, tensors.stream));

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  HANDLE_ERROR(cudaMemcpy(ea_emissivity, ea_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));

  return "CUTENSOR,EA_EMISSIVITY," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
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

  float pos0 = 0;
  float pos1 = 1;
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_div, (void *)&k1, enb_d, (void *)&pos1, radiance_termal_d, tensor_aux1_d, tensors.stream));
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add, (void *)&pos1, tensor_aux1_d, (void *)&pos1, only1_d, tensor_aux1_d, tensors.stream));
  HANDLE_CUTENSOR_ERROR(cutensorPermute(tensors.handle, tensors.tensor_plan_permute_log, (void *)&pos1, tensor_aux1_d, tensor_aux1_d, tensors.stream));
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_div, (void *)&k2, only1_d, (void *)&pos1, tensor_aux1_d, surface_temperature_d, tensors.stream));
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_max, (void *)&pos1, surface_temperature_d, (void *)&pos0, only1_d, surface_temperature_d, tensors.stream));

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  HANDLE_ERROR(cudaMemcpy(surface_temperature, surface_temperature_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));

  return "CUTENSOR,SURFACE_TEMPERATURE," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::short_wave_radiation_function(MTL mtl)
{
  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  float divmtl2 = 1 / (mtl.distance_earth_sun * mtl.distance_earth_sun);
  float costheta = sin(mtl.sun_elevation * PI / 180);
  float cos1367 = 1367 * costheta;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_mult, (void *)&cos1367, tal_d, (void *)&divmtl2, only1_d, short_wave_radiation_d, tensors.stream));

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  HANDLE_ERROR(cudaMemcpy(short_wave_radiation, short_wave_radiation_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));

  return "CUTENSOR,SHORT_WAVE_RADIATION," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::large_wave_radiation_surface_function()
{
  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  float pos1 = 1;
  float pos567 = 5.67;
  float pos1e8 = 1e-8;
  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_mult, (void *)&pos1, surface_temperature_d, (void *)&pos1, surface_temperature_d, tensor_aux1_d, tensors.stream));
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_mult, (void *)&pos1, tensor_aux1_d, (void *)&pos1, tensor_aux1_d, tensor_aux1_d, tensors.stream));
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_mult, (void *)&pos567, eo_d, (void *)&pos1e8, tensor_aux1_d, large_wave_radiation_surface_d, tensors.stream));

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  HANDLE_ERROR(cudaMemcpy(large_wave_radiation_surface, large_wave_radiation_surface_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));

  return "CUTENSOR,LARGE_WAVE_RADIATION_SURFACE," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::large_wave_radiation_atmosphere_function(float temperature)
{
  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  float pos567_1e8 = 5.67 * 1e-8;
  float temperature_kelvin = temperature + 273.15;
  float temperature_kelvin_pow_4 = temperature_kelvin * temperature_kelvin * temperature_kelvin * temperature_kelvin;
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_mult, (void *)&pos567_1e8, ea_d, (void *)&temperature_kelvin_pow_4, only1_d, large_wave_radiation_atmosphere_d, tensors.stream));

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  HANDLE_ERROR(cudaMemcpy(large_wave_radiation_atmosphere, large_wave_radiation_atmosphere_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));

  return "CUTENSOR,LARGE_WAVE_RADIATION_ATMOSPHERE," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::net_radiation_function()
{
  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  float pos0 = 0;
  float pos1 = 1;
  float neg1 = -1;
  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  HANDLE_CUTENSOR_ERROR(cutensorElementwiseTrinaryExecute(tensors.handle, tensors.tensor_plan_trinity_mult_add, (void *)&neg1, short_wave_radiation_d, (void *)&pos1, albedo_d, (void *)&pos1, short_wave_radiation_d, tensor_aux1_d, tensors.stream));
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add, (void *)&pos1, tensor_aux1_d, (void *)&pos1, large_wave_radiation_atmosphere_d, tensor_aux1_d, tensors.stream));
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add, (void *)&pos1, tensor_aux1_d, (void *)&neg1, large_wave_radiation_surface_d, tensor_aux1_d, tensors.stream));
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseTrinaryExecute(tensors.handle, tensors.tensor_plan_trinity_add_mult, (void *)&pos1, only1_d, (void *)&neg1, eo_d, (void *)&pos1, large_wave_radiation_atmosphere_d, tensor_aux2_d, tensors.stream));
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add, (void *)&pos1, tensor_aux1_d, (void *)&neg1, tensor_aux2_d, net_radiation_d, tensors.stream));
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_max, (void *)&pos1, net_radiation_d, (void *)&pos0, only1_d, net_radiation_d, tensors.stream));

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  HANDLE_ERROR(cudaMemcpy(net_radiation, net_radiation_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));

  return "CUTENSOR,NET_RADIATION," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::soil_heat_flux_function()
{
  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  float pos0 = 0;
  float pos1 = 1;
  float pos0074 = 0.0074;
  float post0038 = 0.0038;
  float neg27315 = -273.15;
  float neg098 = -0.98;
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add, (void *)&post0038, only1_d, (void *)&pos0074, albedo_d, tensor_aux1_d, tensors.stream));
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseTrinaryExecute(tensors.handle, tensors.tensor_plan_trinity_add_mult, (void *)&pos1, surface_temperature_d, (void *)&neg27315, only1_d, (void *)&pos1, tensor_aux1_d, tensor_aux1_d, tensors.stream));
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_mult, (void *)&pos1, ndvi_d, (void *)&pos1, ndvi_d, tensor_aux2_d, tensors.stream));
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_mult, (void *)&pos1, tensor_aux2_d, (void *)&pos1, tensor_aux2_d, tensor_aux2_d, tensors.stream));
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add, (void *)&pos1, only1_d, (void *)&neg098, tensor_aux2_d, tensor_aux2_d, tensors.stream));
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_mult, (void *)&pos1, tensor_aux1_d, (void *)&pos1, tensor_aux2_d, tensor_aux1_d, tensors.stream));
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_mult, (void *)&pos1, tensor_aux1_d, (void *)&pos1, net_radiation_d, soil_heat_d, tensors.stream));
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_max, (void *)&pos1, soil_heat_d, (void *)&pos0, only1_d, soil_heat_d, tensors.stream));
  // TODO: treat the values -> soil_heat < 0 and ndvi == 0

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  HANDLE_ERROR(cudaMemcpy(soil_heat, soil_heat_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));

  return "CUTENSOR,SOIL_HEAT_FLUX," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::d0_fuction()
{
  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  float CD1 = 20.6;
  float HGHT = 4;
  float pos1 = 1;
  float neg1 = -1;

  // tensor_aux1_d = CD1 * this->pai[i]
  HANDLE_CUTENSOR_ERROR(cutensorPermute(tensors.handle, tensors.tensor_plan_permute_id, (void *)&CD1, pai_d, tensor_aux1_d, tensors.stream));

  // tensor_aux1_d = sqrt(CD1 * this->pai[i])
  HANDLE_CUTENSOR_ERROR(cutensorPermute(tensors.handle, tensors.tensor_plan_permute_sqtr, (void *)&pos1, tensor_aux1_d, tensor_aux1_d, tensors.stream));

  // tensor_aux2_d = exp(1.0)
  HANDLE_CUTENSOR_ERROR(cutensorPermute(tensors.handle, tensors.tensor_plan_permute_exp, (void *)&pos1, only1_d, tensor_aux2_d, tensors.stream));

  // tensor_aux2_d = -(CD1 * this->pai[i]) * log(exp(1.0))
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_log_mul, (void *)&neg1, tensor_aux1_d, (void *)&pos1, tensor_aux2_d, tensor_aux2_d, tensors.stream));

  // tensor_aux1_d = 1 / cd1_pai_root
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_div, (void *)&pos1, only1_d, (void *)&pos1, tensor_aux1_d, tensor_aux1_d, tensors.stream));

  // tensor_aux2_d  = (1 / cd1_pai_root) * exp(-cd1_pai_root * log(exp(1.0))) ~ pow(exp(1.0), -cd1_pai_root) / cd1_pai_root
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_exp_mul, (void *)&pos1, tensor_aux1_d, (void *)&pos1, tensor_aux2_d, tensor_aux2_d, tensors.stream));

  // tensor_aux1_d = 1 - (1 / cd1_pai_root)
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add, (void *)&pos1, only1_d, (void *)&neg1, tensor_aux1_d, tensor_aux1_d, tensors.stream));

  // d0_d = HGHT * ((1 - (1 / cd1_pai_root)) + (pow(exp(1.0), -cd1_pai_root) / cd1_pai_root))
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add, (void *)&HGHT, tensor_aux1_d, (void *)&pos1, tensor_aux2_d, d0_d, tensors.stream));

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  HANDLE_ERROR(cudaMemcpy(d0, d0_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));

  return "CUTENSOR,D0," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::zom_fuction(float A_ZOM, float B_ZOM)
{
  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  float HGHT = 4;
  float CD = 0.01;
  float CR = 0.35;
  float PSICORR = 0.2;
  float CR2 = CR * 2;
  float neg05 = -0.5;
  float pos1 = 1;
  float pos33 = 3.3;
  float negVON = -VON_KARMAN;
  float neg1 = -1;

  // tensor_aux1_d = (CD + CR2 * this->pai_vector[line][col])
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add, (void *)&CD, only1_d, (void *)&CR2, pai_d, tensor_aux1_d, tensors.stream));

  // tensor_aux1_d = pow(tensor_aux1_d, -0.5); ~ exp(−0.5⋅log(tensor_aux1_d))
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_log_mul, (void *)&neg05, only1_d, (void *)&pos1, tensor_aux1_d, tensor_aux1_d, tensors.stream));
  HANDLE_CUTENSOR_ERROR(cutensorPermute(tensors.handle, tensors.tensor_plan_permute_exp, (void *)&pos1, tensor_aux1_d, tensor_aux1_d, tensors.stream));

  // if (gama < 3.3) gama = 3.3;
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_max, (void *)&pos33, only1_d, (void *)&pos1, tensor_aux1_d, tensor_aux1_d, tensors.stream));

  // tensor_aux1_d = (-VON_KARMAN * gama) + PSICORR
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add, (void *)&negVON, tensor_aux1_d, (void *)&PSICORR, only1_d, tensor_aux1_d, tensors.stream));

  // tensor_aux1_d = pow(exp(1.0), tensor_aux1_d) ~ exp(tensor_aux1_d)
  HANDLE_CUTENSOR_ERROR(cutensorPermute(tensors.handle, tensors.tensor_plan_permute_exp, (void *)&pos1, tensor_aux1_d, tensor_aux1_d, tensors.stream));

  // tensor_aux2_d = HGHT - this->d0_pointer[pos]
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add, (void *)&HGHT, only1_d, (void *)&neg1, d0_d, tensor_aux2_d, tensors.stream));

  // zom = tensor_aux1_d * tensor_aux2_d
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_mult, (void *)&pos1, tensor_aux2_d, (void *)&pos1, tensor_aux1_d, zom_d, tensors.stream));

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  HANDLE_ERROR(cudaMemcpy(zom, zom_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));

  return "CUTENSOR,ZOM," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::ustar_fuction(float u10)
{
  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  float zu = 10;
  float pos1 = 1;
  float neg1 = -1;
  float uVON = u10 * VON_KARMAN;

  // tensor_aux1_d = (zu - DISP)
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add, (void *)&zu, only1_d, (void *)&neg1, d0_d, tensor_aux1_d, tensors.stream));

  // tensor_aux1_d = (zu - DISP) / zom
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_div, (void *)&pos1, tensor_aux1_d, (void *)&pos1, zom_d, tensor_aux1_d, tensors.stream));

  // tensor_aux1_d = log((zu - DISP) / zom)
  HANDLE_CUTENSOR_ERROR(cutensorPermute(tensors.handle, tensors.tensor_plan_permute_log, (void *)&pos1, tensor_aux1_d, tensor_aux1_d, tensors.stream));

  // ustar[i] = (u10 * VON_KARMAN) / log((zu - DISP) / zom);
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_div, (void *)&uVON, only1_d, (void *)&pos1, tensor_aux1_d, ustar_d, tensors.stream));

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  HANDLE_ERROR(cudaMemcpy(ustar, ustar_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));

  return "CUTENSOR,USTAR," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
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

  float pos1 = 1;
  float pos2 = 2;
  float pos025 = 0.25;
  float pos246 = 2.46;
  float pos009 = 0.009;
  float pos4631 = 0.4631;
  float neg05 = -0.5;
  float neg1 = -1;
  float neg2 = -2;

  float ct4= 4 * ct;
  float cdc3 = cd * -c3;
  float divHGHT = 1 / HGHT;
  float div_visc = 1 / visc;
  float cdVON = cd * VON_KARMAN;
  float pow_pr = pow(pr, -0.667);
  float neg_ndvi_max = -ndvi_max;
  float soil_moisture_day_rel = 0.33;
  float div_ndvi_min_max = 1 / (ndvi_min - ndvi_max);
  float SF = sf_c + (1 / (1 + pow(exp(1.0), (sf_d - (sf_e * soil_moisture_day_rel)))));

  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  // float Re_star = (this->ustar[i] * 0.009) / visc;
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_mult,
                                                         (void *)&pos009, ustar_d,
                                                         (void *)&div_visc, only1_d,
                                                         Re_star_d, tensors.stream));

  // float Ct_star = pow(pr, -0.667) * pow(Re_star, -0.5); ~ pow_pr * exp(-0.5 * log(Re_star)); // 0.02%
  HANDLE_CUTENSOR_ERROR(cutensorPermute(tensors.handle, tensors.tensor_plan_permute_log, (void *)&neg05, Re_star_d, Ct_star_d, tensors.stream));
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_exp_mul,
                                                         (void *)&pow_pr, only1_d,
                                                         (void *)&pos1, Ct_star_d,
                                                         Ct_star_d, tensors.stream));

  // float beta = c1 - c2 * (exp(cdc3 * this->pai[i])); // OK
  HANDLE_CUTENSOR_ERROR(cutensorPermute(tensors.handle, tensors.tensor_plan_permute_id, (void *)&cdc3, pai_d, beta_d, tensors.stream));
  HANDLE_CUTENSOR_ERROR(cutensorPermute(tensors.handle, tensors.tensor_plan_permute_exp, (void *)&c2, beta_d, beta_d, tensors.stream));
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add,
                                                         (void *)&c1, only1_d,
                                                         (void *)&neg1, beta_d,
                                                         beta_d, tensors.stream));

  // float nec_terra = (cd * this->pai[i]) / (beta * beta * 2);  // OK
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_mult,
                                                         (void *)&pos2, beta_d,
                                                         (void *)&pos1, beta_d,
                                                         nec_terra_d, tensors.stream));
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_div,
                                                         (void *)&cd, pai_d,
                                                         (void *)&pos1, nec_terra_d,
                                                         nec_terra_d, tensors.stream));

  // float kb1_fst_part = (cd * VON_KARMAN) / (4 * ct * beta * (1 - exp(nec_terra * -0.5)));
  HANDLE_CUTENSOR_ERROR(cutensorPermute(tensors.handle, tensors.tensor_plan_permute_id, (void *)&neg05, nec_terra_d, kb1_fst_part_d, tensors.stream));
  HANDLE_CUTENSOR_ERROR(cutensorPermute(tensors.handle, tensors.tensor_plan_permute_exp, (void *)&pos1, kb1_fst_part_d, kb1_fst_part_d, tensors.stream));
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add,
                                                         (void *)&pos1, only1_d,
                                                         (void *)&neg1, kb1_fst_part_d,
                                                         kb1_fst_part_d, tensors.stream));
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_mult,
                                                         (void *)&ct4, beta_d,
                                                         (void *)&pos1, kb1_fst_part_d,
                                                         kb1_fst_part_d, tensors.stream));
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_div,
                                                         (void *)&cdVON, only1_d,
                                                         (void *)&pos1, kb1_fst_part_d,
                                                         kb1_fst_part_d, tensors.stream));

  // float kb1_sec_part = (beta * VON_KARMAN * (this->zom[i] / HGHT)) / Ct_star;
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_mult,
                                                         (void *)&divHGHT, only1_d,
                                                         (void *)&pos1, zom_d,
                                                         kb1_sec_part_d, tensors.stream));
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_mult,
                                                         (void *)&VON_KARMAN, beta_d,
                                                         (void *)&pos1, kb1_sec_part_d,
                                                         kb1_sec_part_d, tensors.stream));
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_div,
                                                         (void *)&pos1, kb1_sec_part_d,
                                                         (void *)&pos1, Ct_star_d,
                                                         kb1_sec_part_d, tensors.stream));

  // float kb1s = (pow(Re_star, 0.25) * 2.46) - 2) ~ 2.46 * exp(0.25 * log(Re_star));
  HANDLE_CUTENSOR_ERROR(cutensorPermute(tensors.handle, tensors.tensor_plan_permute_log, (void *)&pos025, Re_star_d, kb1s_d, tensors.stream));
  HANDLE_CUTENSOR_ERROR(cutensorPermute(tensors.handle, tensors.tensor_plan_permute_exp, (void *)&pos246, kb1s_d, kb1s_d, tensors.stream));
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add,
                                                         (void *)&pos1, kb1s_d,
                                                         (void *)&neg2, only1_d,
                                                         kb1s_d, tensors.stream));

  // float fc = 1 - pow((this->ndvi[i] - ndvi_max) / (ndvi_min - ndvi_max), 0.4631);
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add,
                                                         (void *)&pos1, ndvi_d,
                                                         (void *)&neg_ndvi_max, only1_d,
                                                         fc_d, tensors.stream));
  HANDLE_CUTENSOR_ERROR(cutensorPermute(tensors.handle, tensors.tensor_plan_permute_id, (void *)&div_ndvi_min_max, fc_d, fc_d, tensors.stream));
  HANDLE_CUTENSOR_ERROR(cutensorPermute(tensors.handle, tensors.tensor_plan_permute_log, (void *)&pos4631, fc_d, fc_d, tensors.stream));
  HANDLE_CUTENSOR_ERROR(cutensorPermute(tensors.handle, tensors.tensor_plan_permute_exp, (void *)&pos1, fc_d, fc_d, tensors.stream));
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add,
                                                         (void *)&pos1, only1_d,
                                                         (void *)&neg1, fc_d,
                                                         fc_d, tensors.stream));

  // float fs = 1 - fc;
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add,
                                                         (void *)&pos1, only1_d,
                                                         (void *)&neg1, fc_d,
                                                         fs_d, tensors.stream));

  // this->kb1[i] = ((kb1_fst_part * pow(fc, 2)) + (kb1_sec_part * pow(fc, 2) * pow(fs, 2)) + (pow(fs, 2) * kb1s)) * SF;
  HANDLE_CUTENSOR_ERROR(cutensorPermute(tensors.handle, tensors.tensor_plan_permute_log, (void *)&pos2, fc_d, fcpow_d, tensors.stream));
  HANDLE_CUTENSOR_ERROR(cutensorPermute(tensors.handle, tensors.tensor_plan_permute_exp, (void *)&pos1, fcpow_d, fcpow_d, tensors.stream));
  HANDLE_CUTENSOR_ERROR(cutensorPermute(tensors.handle, tensors.tensor_plan_permute_log, (void *)&pos2, fs_d, fspow_d, tensors.stream));
  HANDLE_CUTENSOR_ERROR(cutensorPermute(tensors.handle, tensors.tensor_plan_permute_exp, (void *)&pos1, fspow_d, fspow_d, tensors.stream));
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_mult,
                                                         (void *)&pos1, kb1_fst_part_d,
                                                         (void *)&pos1, fcpow_d,
                                                         tensor_aux1_d, tensors.stream));
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_mult,
                                                         (void *)&pos1, fcpow_d,
                                                         (void *)&pos1, fspow_d,
                                                         tensor_aux2_d, tensors.stream));
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_mult,
                                                         (void *)&pos1, kb1_sec_part_d,
                                                         (void *)&pos1, tensor_aux2_d,
                                                         tensor_aux2_d, tensors.stream));
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_mult,
                                                         (void *)&pos1, kb1s_d,
                                                         (void *)&pos1, fspow_d,
                                                         kb1_d, tensors.stream));
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add,
                                                         (void *)&pos1, kb1_d,
                                                         (void *)&pos1, tensor_aux2_d,
                                                         kb1_d, tensors.stream));
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add,
                                                         (void *)&pos1, kb1_d,
                                                         (void *)&pos1, tensor_aux1_d,
                                                         kb1_d, tensors.stream));
  HANDLE_CUTENSOR_ERROR(cutensorPermute(tensors.handle, tensors.tensor_plan_permute_id, (void *)&SF, kb1_d, kb1_d, tensors.stream));

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  HANDLE_ERROR(cudaMemcpy(kb1, kb1_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));

  return "CUDACORE,KB," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::aerodynamic_resistance_fuction()
{
  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  aerodynamic_resistance_kernel<<<this->blocks_num, this->threads_num>>>(zom_d, d0_d, ustar_d, kb1_d, rah_d, width_band, height_band);

  HANDLE_ERROR(cudaDeviceSynchronize());
  HANDLE_ERROR(cudaGetLastError());

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  HANDLE_ERROR(cudaMemcpy(aerodynamic_resistance, rah_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));

  return "CUDACORE,RAH_INI," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::sensible_heat_flux_function(float a, float b)
{
  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  float pos1 = 1;
  float neg1 = -1;
  float neg27315 = -273.15;
  float RHO_AIR = RHO * SPECIFIC_HEAT_AIR;

  // tensor_aux1_d = this->surface_temperature[i] - 273.15
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add,
                                                         (void *)&pos1, surface_temperature_d, (void *)&neg27315, only1_d, tensor_aux1_d, tensors.stream));

  // tensor_aux1_d = a + b * (this->surface_temperature[i] - 273.15))
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add,
                                                         (void *)&a, only1_d, (void *)&b, tensor_aux1_d, tensor_aux1_d, tensors.stream));

  // sensible_heat_flux[i] = RHO * SPECIFIC_HEAT_AIR * (a + b * (this->surface_temperature[i] - 273.15)) / this->aerodynamic_resistance[i];
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_div,
                                                         (void *)&RHO_AIR, tensor_aux1_d, (void *)&pos1, rah_d, sensible_heat_flux_d, tensors.stream));

  // tensor_aux2_d = this->net_radiation[i] - this->soil_heat[i]
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add,
                                                         (void *)&pos1, net_radiation_d, (void *)&neg1, soil_heat_d, tensor_aux2_d, tensors.stream));

  // if (sensible_heat_flux[i] > (this->net_radiation[i] - this->soil_heat[i])) sensible_heat_flux[i] = this->net_radiation[i] - this->soil_heat[i];
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_min,
                                                         (void *)&pos1, sensible_heat_flux_d, (void *)&pos1, tensor_aux2_d, sensible_heat_flux_d, tensors.stream));

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  HANDLE_ERROR(cudaMemcpy(sensible_heat_flux, sensible_heat_flux_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));

  return "CUTENSOR,SENSIBLE_HEAT_FLUX," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::latent_heat_flux_function()
{
  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  float pos1 = 1;
  float neg1 = -1;

  // this->latent_heat_flux[i] = this->net_radiation[i] - this->soil_heat[i] - this->sensible_heat_flux[i];
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add, (void *)&pos1, net_radiation_d, (void *)&neg1, soil_heat_d, tensor_aux1_d, tensors.stream));
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add, (void *)&pos1, tensor_aux1_d, (void *)&neg1, sensible_heat_flux_d, latent_heat_flux_d, tensors.stream));

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  HANDLE_ERROR(cudaMemcpy(latent_heat_flux, latent_heat_flux_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));

  return "CUTENSOR,LATENT_HEAT_FLUX," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::net_radiation_24h_function(float Ra24h, float Rs24h)
{
  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  int FL = 110;
  float pos1 = 1;
  float neg1 = -1;
  float negFLRsRa = -(FL * Rs24h / Ra24h);

  // this->net_radiation_24h[i] = (1 - this->albedo[i]) * Rs24h - FL * Rs24h / Ra24h;
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add, (void *)&pos1, only1_d, (void *)&neg1, albedo_d, tensor_aux1_d, tensors.stream));
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add, (void *)&Rs24h, tensor_aux1_d, (void *)&negFLRsRa, only1_d, net_radiation_24h_d, tensors.stream));

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  HANDLE_ERROR(cudaMemcpy(net_radiation_24h, net_radiation_24h_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));

  return "CUTENSOR,NET_RADIATION_24H," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::evapotranspiration_fraction_fuction()
{
  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  float pos1 = 1;
  float neg1 = -1;

  // this->evapotranspiration_fraction[i] = this->latent_heat_flux[i] / (this->net_radiation[i] - this->soil_heat[i]);
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add, (void *)&pos1, net_radiation_d, (void *)&neg1, soil_heat_d, tensor_aux1_d, tensors.stream));
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_div, (void *)&pos1, latent_heat_flux_d, (void *)&pos1, tensor_aux1_d, evapotranspiration_fraction_d, tensors.stream));

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  HANDLE_ERROR(cudaMemcpy(evapotranspiration_fraction, evapotranspiration_fraction_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));

  return "CUTENSOR,EVAPOTRANSPIRATION_FRACTION," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::sensible_heat_flux_24h_fuction()
{
  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  float pos1 = 1;
  float neg1 = -1;

  // this->sensible_heat_flux_24h[i] = (1 - this->evapotranspiration_fraction[i]) * this->net_radiation_24h[i];
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_add, (void *)&pos1, only1_d, (void *)&neg1, evapotranspiration_fraction_d, tensor_aux1_d, tensors.stream));
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_mult, (void *)&pos1, tensor_aux1_d, (void *)&pos1, net_radiation_24h_d, sensible_heat_flux_24h_d, tensors.stream));

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  HANDLE_ERROR(cudaMemcpy(sensible_heat_flux_24h, sensible_heat_flux_24h_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));

  return "CUTENSOR,SENSIBLE_HEAT_FLUX_24H," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::latent_heat_flux_24h_function()
{
  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  float pos1 = 1;

  // this->latent_heat_flux_24h[i] = this->evapotranspiration_fraction[i] * this->net_radiation_24h[i];
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_mult, (void *)&pos1, evapotranspiration_fraction_d, (void *)&pos1, net_radiation_24h_d, latent_heat_flux_24h_d, tensors.stream));

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  HANDLE_ERROR(cudaMemcpy(latent_heat_flux_24h, latent_heat_flux_24h_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));

  return "CUTENSOR,LATENT_HEAT_FLUX_24H," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::evapotranspiration_24h_function(Station station)
{
  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  float pos86400 = 86400;
  float div = 1 / ((2.501 - 0.00236 * (station.v7_max + station.v7_min) / 2) * 1e+6);

  // this->evapotranspiration_24h[i] = (this->latent_heat_flux_24h[i] * 86400) / ((2.501 - 0.00236 * (station.v7_max + station.v7_min) / 2) * 1e+6);
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_mult, (void *)&pos86400, latent_heat_flux_24h_d, (void *)&div, only1_d, evapotranspiration_24h_d, tensors.stream));

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  HANDLE_ERROR(cudaMemcpy(evapotranspiration_24h, evapotranspiration_24h_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));

  return "CUTENSOR,EVAPOTRANSPIRATION_24H," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::evapotranspiration_function()
{
  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  float pos1 = 1;
  float pos0035 = 0.035;

  // this->evapotranspiration[i] = this->net_radiation_24h[i] * this->evapotranspiration_fraction[i] * 0.035;
  HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinaryExecute(tensors.handle, tensors.tensor_plan_binary_mult, (void *)&pos1, net_radiation_24h_d, (void *)&pos0035, evapotranspiration_fraction_d, evapotranspiration_d, tensors.stream));

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  HANDLE_ERROR(cudaMemcpy(evapotranspiration, evapotranspiration_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));

  return "CUTENSOR,EVAPOTRANSPIRATION," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::rah_correction_function_blocks(float ndvi_min, float ndvi_max, Candidate hot_pixel, Candidate cold_pixel)
{
  system_clock::time_point begin_core, end_core;
  int64_t general_time_core, initial_time_core, final_time_core;

  // ========= CUDA Setup
  int dev = 0;
  cudaDeviceProp deviceProp;
  HANDLE_ERROR(cudaGetDeviceProperties(&deviceProp, dev));
  HANDLE_ERROR(cudaSetDevice(dev));

  int threads_per_block = threads_num;
  int num_blocks = ceil(width_band * height_band / threads_per_block);

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

    rah_correction_cycle_STEEP<<<num_blocks, threads_per_block>>>(surface_temperature_d, d0_d, kb1_d, zom_d, ustar_d, rah_d, sensible_heat_flux_d, a, b, height_band, width_band);
    HANDLE_ERROR(cudaDeviceSynchronize());
    HANDLE_ERROR(cudaGetLastError());

    end_core = system_clock::now();
    general_time_core = duration_cast<nanoseconds>(end_core - begin_core).count();
    final_time_core = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
    // ====

    HANDLE_ERROR(cudaMemcpy(ustar, ustar_d, nBytes_band, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(aerodynamic_resistance, rah_d, nBytes_band, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(sensible_heat_flux, sensible_heat_flux_d, nBytes_band, cudaMemcpyDeviceToHost));

    float rah_hot = this->aerodynamic_resistance[hot_pixel.line * width_band + hot_pixel.col];
    hot_pixel.setAerodynamicResistance(rah_hot);

    float rah_cold = this->aerodynamic_resistance[cold_pixel.line * width_band + cold_pixel.col];
    cold_pixel.setAerodynamicResistance(rah_cold);
  }

  return "CUDACORE,RAH_CYCLE," + std::to_string(general_time_core) + "," + std::to_string(initial_time_core) + "," + std::to_string(final_time_core) + "\n";
}
