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
  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  rad_kernel<<<this->blocks_num, this->threads_num>>>(band_blue_d, band_green_d, band_red_d, band_nir_d, band_swir1_d, band_termal_d, band_swir2_d,
                                                      radiance_blue_d, radiance_green_d, radiance_red_d, radiance_nir_d, radiance_swir1_d, radiance_termal_d, radiance_swir2_d,
                                                      mtl.rad_add_d, mtl.rad_mult_d, width_band, height_band);
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

  return "CUDACORE,RADIANCE," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
}

string Products::reflectance_function(MTL mtl)
{
  const float sin_sun = sin(mtl.sun_elevation * PI / 180);

  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  ref_kernel<<<this->blocks_num, this->threads_num>>>(band_blue_d, band_green_d, band_red_d, band_nir_d, band_swir1_d, band_termal_d, band_swir2_d,
                                                      reflectance_blue_d, reflectance_green_d, reflectance_red_d, reflectance_nir_d, reflectance_swir1_d, reflectance_termal_d, reflectance_swir2_d,
                                                      mtl.ref_add_d, mtl.ref_mult_d, sin_sun, width_band, height_band);

  HANDLE_ERROR(cudaDeviceSynchronize());
  HANDLE_ERROR(cudaGetLastError());

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

  return "CUDACORE,REFLECTANCE," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
}

string Products::albedo_function(MTL mtl)
{
  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  albedo_kernel<<<this->blocks_num, this->threads_num>>>(reflectance_blue_d, reflectance_green_d, reflectance_red_d, reflectance_nir_d, reflectance_swir1_d, reflectance_swir2_d,
                                                         tal_d, albedo_d, mtl.ref_w_coeff_d, width_band, height_band);

  HANDLE_ERROR(cudaDeviceSynchronize());
  HANDLE_ERROR(cudaGetLastError());

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  HANDLE_ERROR(cudaMemcpy(albedo, albedo_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));

  return "CUDACORE,ALBEDO," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
}

string Products::ndvi_function()
{
  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  ndvi_kernel<<<this->blocks_num, this->threads_num>>>(reflectance_nir_d, reflectance_red_d, ndvi_d, width_band, height_band);

  HANDLE_ERROR(cudaDeviceSynchronize());
  HANDLE_ERROR(cudaGetLastError());

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  HANDLE_ERROR(cudaMemcpy(ndvi, ndvi_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));

  return "CUDACORE,NDVI," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::pai_function()
{
  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  pai_kernel<<<this->blocks_num, this->threads_num>>>(reflectance_nir_d, reflectance_red_d, pai_d, width_band, height_band);

  HANDLE_ERROR(cudaDeviceSynchronize());
  HANDLE_ERROR(cudaGetLastError());

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  HANDLE_ERROR(cudaMemcpy(pai, pai_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));

  return "CUDACORE,PAI," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::lai_function()
{
  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

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

  evi_kernel<<<this->blocks_num, this->threads_num>>>(reflectance_nir_d, reflectance_red_d, reflectance_blue_d, evi_d, width_band, height_band);

  HANDLE_ERROR(cudaDeviceSynchronize());
  HANDLE_ERROR(cudaGetLastError());

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  HANDLE_ERROR(cudaMemcpy(evi, evi_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));

  return "CUDACORE,EVI," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::enb_emissivity_function()
{
  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

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

  ea_kernel<<<this->blocks_num, this->threads_num>>>(tal_d, ea_d, width_band, height_band);

  HANDLE_ERROR(cudaDeviceSynchronize());
  HANDLE_ERROR(cudaGetLastError());

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  HANDLE_ERROR(cudaMemcpy(ea_emissivity, ea_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));

  return "CUDACORE,EA_EMISSIVITY," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
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

  surface_temperature_kernel<<<this->blocks_num, this->threads_num>>>(enb_d, radiance_termal_d, surface_temperature_d, k1, k2, width_band, height_band);

  HANDLE_ERROR(cudaDeviceSynchronize());
  HANDLE_ERROR(cudaGetLastError());

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  HANDLE_ERROR(cudaMemcpy(surface_temperature, surface_temperature_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));

  return "CUDACORE,SURFACE_TEMPERATURE," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::short_wave_radiation_function(MTL mtl)
{
  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  short_wave_radiation_kernel<<<this->blocks_num, this->threads_num>>>(tal_d, short_wave_radiation_d, mtl.sun_elevation, mtl.distance_earth_sun, PI, width_band, height_band);

  HANDLE_ERROR(cudaDeviceSynchronize());
  HANDLE_ERROR(cudaGetLastError());

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  HANDLE_ERROR(cudaMemcpy(short_wave_radiation, short_wave_radiation_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));

  return "CUDACORE,SHORT_WAVE_RADIATION," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::large_wave_radiation_surface_function()
{
  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  large_wave_radiation_surface_kernel<<<this->blocks_num, this->threads_num>>>(surface_temperature_d, eo_d, large_wave_radiation_surface_d, width_band, height_band);

  HANDLE_ERROR(cudaDeviceSynchronize());
  HANDLE_ERROR(cudaGetLastError());

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  HANDLE_ERROR(cudaMemcpy(large_wave_radiation_surface, large_wave_radiation_surface_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));

  return "CUDACORE,LARGE_WAVE_RADIATION_SURFACE," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::large_wave_radiation_atmosphere_function(float temperature)
{
  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  large_wave_radiation_atmosphere_kernel<<<this->blocks_num, this->threads_num>>>(ea_d, large_wave_radiation_atmosphere_d, temperature, width_band, height_band);

  HANDLE_ERROR(cudaDeviceSynchronize());
  HANDLE_ERROR(cudaGetLastError());

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  HANDLE_ERROR(cudaMemcpy(large_wave_radiation_atmosphere, large_wave_radiation_atmosphere_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));

  return "CUDACORE,LARGE_WAVE_RADIATION_ATMOSPHERE," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::net_radiation_function()
{
  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  net_radiation_kernel<<<this->blocks_num, this->threads_num>>>(short_wave_radiation_d, albedo_d, large_wave_radiation_atmosphere_d, large_wave_radiation_surface_d, eo_d, net_radiation_d, width_band, height_band);

  HANDLE_ERROR(cudaDeviceSynchronize());
  HANDLE_ERROR(cudaGetLastError());

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  HANDLE_ERROR(cudaMemcpy(net_radiation, net_radiation_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));

  return "CUDACORE,NET_RADIATION," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::soil_heat_flux_function()
{
  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  soil_heat_kernel<<<this->blocks_num, this->threads_num>>>(ndvi_d, albedo_d, surface_temperature_d, net_radiation_d, soil_heat_d, width_band, height_band);

  HANDLE_ERROR(cudaDeviceSynchronize());
  HANDLE_ERROR(cudaGetLastError());

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  HANDLE_ERROR(cudaMemcpy(soil_heat, soil_heat_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));

  return "CUDACORE,SOIL_HEAT_FLUX," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::d0_fuction()
{
  float CD1 = 20.6;
  float HGHT = 4;

  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  d0_kernel<<<this->blocks_num, this->threads_num>>>(pai_d, d0_d, CD1, HGHT, width_band, height_band);

  HANDLE_ERROR(cudaDeviceSynchronize());
  HANDLE_ERROR(cudaGetLastError());

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  HANDLE_ERROR(cudaMemcpy(d0, d0_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));

  return "CUDACORE,D0," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::zom_fuction(float A_ZOM, float B_ZOM)
{
  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  zom_kernel<<<this->blocks_num, this->threads_num>>>(d0_d, pai_d, zom_d, A_ZOM, B_ZOM, width_band, height_band);

  HANDLE_ERROR(cudaDeviceSynchronize());
  HANDLE_ERROR(cudaGetLastError());

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  HANDLE_ERROR(cudaMemcpy(zom, zom_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));

  return "CUDACORE,ZOM," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::ustar_fuction(float u10)
{
  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  ustar_kernel<<<this->blocks_num, this->threads_num>>>(zom_d, d0_d, ustar_d, u10, width_band, height_band);

  HANDLE_ERROR(cudaDeviceSynchronize());
  HANDLE_ERROR(cudaGetLastError());

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  HANDLE_ERROR(cudaMemcpy(ustar, ustar_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));

  return "CUDACORE,USTAR," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::kb_function(float ndvi_max, float ndvi_min)
{
  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  kb_kernel<<<this->blocks_num, this->threads_num>>>(zom_d, ustar_d, pai_d, kb1_d, ndvi_d, width_band, height_band, ndvi_max, ndvi_min);

  HANDLE_ERROR(cudaDeviceSynchronize());
  HANDLE_ERROR(cudaGetLastError());

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

  sensible_heat_flux_kernel<<<this->blocks_num, this->threads_num>>>(surface_temperature_d, rah_d, net_radiation_d, soil_heat_d, sensible_heat_flux_d, a, b, width_band, height_band);

  HANDLE_ERROR(cudaDeviceSynchronize());
  HANDLE_ERROR(cudaGetLastError());

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  HANDLE_ERROR(cudaMemcpy(sensible_heat_flux, sensible_heat_flux_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));

  return "CUDACORE,SENSIBLE_HEAT_FLUX," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::latent_heat_flux_function()
{
  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  latent_heat_flux_kernel<<<this->blocks_num, this->threads_num>>>(net_radiation_d, soil_heat_d, sensible_heat_flux_d, latent_heat_flux_d, width_band, height_band);

  HANDLE_ERROR(cudaDeviceSynchronize());
  HANDLE_ERROR(cudaGetLastError());

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  HANDLE_ERROR(cudaMemcpy(latent_heat_flux, latent_heat_flux_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));

  return "CUDACORE,LATENT_HEAT_FLUX," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::net_radiation_24h_function(float Ra24h, float Rs24h)
{
  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  net_radiation_24h_kernel<<<this->blocks_num, this->threads_num>>>(albedo_d, Rs24h, Ra24h, net_radiation_24h_d, width_band, height_band);

  HANDLE_ERROR(cudaDeviceSynchronize());
  HANDLE_ERROR(cudaGetLastError());

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  HANDLE_ERROR(cudaMemcpy(net_radiation_24h, net_radiation_24h_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));

  return "CUDACORE,NET_RADIATION_24H," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::evapotranspiration_fraction_fuction()
{
  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  evapotranspiration_fraction_kernel<<<this->blocks_num, this->threads_num>>>(net_radiation_d, soil_heat_d, latent_heat_flux_d, evapotranspiration_fraction_d, width_band, height_band);

  HANDLE_ERROR(cudaDeviceSynchronize());
  HANDLE_ERROR(cudaGetLastError());

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  HANDLE_ERROR(cudaMemcpy(evapotranspiration_fraction, evapotranspiration_fraction_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));

  return "CUDACORE,EVAPOTRANSPIRATION_FRACTION," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::sensible_heat_flux_24h_fuction()
{
  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  sensible_heat_flux_24h_kernel<<<this->blocks_num, this->threads_num>>>(net_radiation_24h_d, evapotranspiration_fraction_d, sensible_heat_flux_24h_d, width_band, height_band);

  HANDLE_ERROR(cudaDeviceSynchronize());
  HANDLE_ERROR(cudaGetLastError());

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  HANDLE_ERROR(cudaMemcpy(sensible_heat_flux_24h, sensible_heat_flux_24h_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));

  return "CUDACORE,SENSIBLE_HEAT_FLUX_24H," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::latent_heat_flux_24h_function()
{
  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  latent_heat_flux_24h_kernel<<<this->blocks_num, this->threads_num>>>(net_radiation_24h_d, evapotranspiration_fraction_d, latent_heat_flux_24h_d, width_band, height_band);

  HANDLE_ERROR(cudaDeviceSynchronize());
  HANDLE_ERROR(cudaGetLastError());

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  HANDLE_ERROR(cudaMemcpy(latent_heat_flux_24h, latent_heat_flux_24h_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));

  return "CUDACORE,LATENT_HEAT_FLUX_24H," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::evapotranspiration_24h_function(Station station)
{
  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  evapotranspiration_24h_kernel<<<this->blocks_num, this->threads_num>>>(latent_heat_flux_24h_d, evapotranspiration_24h_d, station.v7_max, station.v7_min, width_band, height_band);

  HANDLE_ERROR(cudaDeviceSynchronize());
  HANDLE_ERROR(cudaGetLastError());

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  HANDLE_ERROR(cudaMemcpy(evapotranspiration_24h, evapotranspiration_24h_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));

  return "CUDACORE,EVAPOTRANSPIRATION_24H," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::evapotranspiration_function()
{
  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  evapotranspiration_kernel<<<this->blocks_num, this->threads_num>>>(net_radiation_24h_d, evapotranspiration_fraction_d, evapotranspiration_d, width_band, height_band);

  HANDLE_ERROR(cudaDeviceSynchronize());
  HANDLE_ERROR(cudaGetLastError());

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  HANDLE_ERROR(cudaMemcpy(evapotranspiration, evapotranspiration_d, sizeof(float) * height_band * width_band, cudaMemcpyDeviceToHost));

  return "CUDACORE,EVAPOTRANSPIRATION," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::rah_correction_function_blocks_STEEP(float ndvi_min, float ndvi_max, Candidate hot_pixel, Candidate cold_pixel)
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

string Products::rah_correction_function_blocks_ASEBAL(float ndvi_min, float ndvi_max, Candidate hot_pixel, Candidate cold_pixel, float u200)
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

  int i = 0;
  while (true)
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

    rah_correction_cycle_ASEBAL<<<num_blocks, threads_per_block>>>(surface_temperature_d, kb1_d, zom_d, ustar_d, rah_d, sensible_heat_flux_d, a, b, u200, height_band, width_band);
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

    if (i > 0 && fabs(1 - rah_ini_pq_terra / rah_hot) < 0.05)
      break;
    else 
      i++;
  }

  return "CUDACORE,RAH_CYCLE," + std::to_string(general_time_core) + "," + std::to_string(initial_time_core) + "," + std::to_string(final_time_core) + "\n";
}