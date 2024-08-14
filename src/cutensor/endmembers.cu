#include "endmembers.h"
#include "filter.cuh"
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

void get_quartiles(float *target, float *v_quartile, int height_band, int width_band, float first_interval, float middle_interval, float last_interval)
{
  const int SIZE = height_band * width_band;
  float *target_values = (float *)malloc(sizeof(float) * SIZE);

  if (target_values == NULL)
    exit(15);

  int pos = 0;
  for (int i = 0; i < height_band * width_band; i++)
  {
    if (!isnan(target[i]) && !isinf(target[i]))
    {
      target_values[pos] = target[i];
      pos++;
    }
  }

  int first_index = static_cast<int>(floor(first_interval * pos));
  int middle_index = static_cast<int>(floor(middle_interval * pos));
  int last_index = static_cast<int>(floor(last_interval * pos));

  std::nth_element(target_values, target_values + first_index, target_values + pos);
  v_quartile[0] = target_values[first_index];

  std::nth_element(target_values, target_values + middle_index, target_values + pos);
  v_quartile[1] = target_values[middle_index];

  std::nth_element(target_values, target_values + last_index, target_values + pos);
  v_quartile[2] = target_values[last_index];

  free(target_values);
}

void get_quartiles_cuda(float *d_target, float *v_quartile, int height_band, int width_band,
                   float first_interval, float middle_interval, float last_interval,
                   int blocks_num, int threads_num)
{
  float *d_filtered;
  cudaMalloc(&d_filtered, sizeof(float) * height_band * width_band);

  int indexes[1] = {0};
  int *d_indexes;
  cudaMalloc((void **)&d_indexes, sizeof(int) * 1);
  cudaMemcpy(d_indexes, indexes, sizeof(int) * 1, cudaMemcpyHostToDevice);

  filter_valid_values<<<blocks_num, threads_num>>>(d_target, d_filtered, height_band, width_band, d_indexes);

  cudaMemcpy(&indexes[0], d_indexes, sizeof(int), cudaMemcpyDeviceToHost);

  // Use Thrust to sort the valid elements on the GPU
  thrust::device_ptr<float> d_filtered_ptr = thrust::device_pointer_cast(d_filtered);
  thrust::sort(thrust::device, d_filtered_ptr, d_filtered_ptr + indexes[0]);

  int first_index = static_cast<int>(floor(first_interval * indexes[0]));
  int middle_index = static_cast<int>(floor(middle_interval * indexes[0]));
  int last_index = static_cast<int>(floor(last_interval * indexes[0]));

  cudaMemcpy(&v_quartile[0], d_filtered + first_index, sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(&v_quartile[1], d_filtered + middle_index, sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(&v_quartile[2], d_filtered + last_index, sizeof(float), cudaMemcpyDeviceToHost);

  // Free GPU memory
  cudaFree(d_filtered);
  cudaFree(d_indexes);
}

pair<Candidate, Candidate> getEndmembersSTEPP(float *ndvi, float *d_ndvi, float *surface_temperature, float *d_surface_temperature, float *albedo, float *d_albedo,
                                              float *net_radiation, float *d_net_radiation, float *soil_heat, float *d_soil_heat,
                                              int blocks_num, int threads_num, int height_band, int width_band)
{
  const size_t MAXC = sizeof(Candidate) * height_band * width_band;

  float *d_ho;
  int *d_indexes;
  Candidate *d_hotCandidates, *d_coldCandidates;
  cudaMalloc((void **)&d_ho, sizeof(float) * height_band * width_band);

  int indexes[2] = {0, 0};
  cudaMalloc((void **)&d_indexes, sizeof(int) * 2);
  cudaMemcpy(d_indexes, indexes, sizeof(int) * 2, cudaMemcpyHostToDevice);

  cudaError_t err;
  err = cudaMalloc((void **)&d_hotCandidates, MAXC);
  if (err != cudaSuccess)
  {
    std::cerr << "CUDA memory allocation for d_hotCandidates failed: " << cudaGetErrorString(err) << std::endl;
    // Handle the error appropriately
  }

  err = cudaMalloc((void **)&d_coldCandidates, MAXC);
  if (err != cudaSuccess)
  {
    std::cerr << "CUDA memory allocation for d_coldCandidates failed: " << cudaGetErrorString(err) << std::endl;
    // Handle the error appropriately
  }

  Candidate *hotCandidates, *coldCandidates;
  hotCandidates = (Candidate *)malloc(MAXC);
  coldCandidates = (Candidate *)malloc(MAXC);

  vector<float> tsQuartile(3);
  vector<float> ndviQuartile(3);
  vector<float> albedoQuartile(3);
  get_quartiles_cuda(d_ndvi, ndviQuartile.data(), height_band, width_band, 0.15, 0.97, 0.97, blocks_num, threads_num);
  get_quartiles_cuda(d_albedo, albedoQuartile.data(), height_band, width_band, 0.25, 0.50, 0.75, blocks_num, threads_num);
  get_quartiles_cuda(d_surface_temperature, tsQuartile.data(), height_band, width_band, 0.20, 0.85, 0.97, blocks_num, threads_num);

  process_pixels<<<blocks_num, threads_num>>>(d_hotCandidates, d_coldCandidates, d_indexes,
                                              d_ndvi, d_surface_temperature, d_albedo, d_net_radiation, d_soil_heat, d_ho,
                                              ndviQuartile[0], ndviQuartile[1], tsQuartile[0], tsQuartile[1], tsQuartile[2],
                                              albedoQuartile[0], albedoQuartile[1], albedoQuartile[2], height_band, width_band);

  HANDLE_ERROR(cudaDeviceSynchronize());
  HANDLE_ERROR(cudaGetLastError());

  cudaMemcpy(&indexes, d_indexes, sizeof(int) * 2, cudaMemcpyDeviceToHost);
  cudaMemcpy(hotCandidates, d_hotCandidates, sizeof(Candidate) * indexes[0], cudaMemcpyDeviceToHost);
  cudaMemcpy(coldCandidates, d_coldCandidates, sizeof(Candidate) * indexes[1], cudaMemcpyDeviceToHost);

  std::sort(hotCandidates, hotCandidates + indexes[0], compare_candidate_temperature);
  std::sort(coldCandidates, coldCandidates + indexes[1], compare_candidate_temperature);

  unsigned int hotPos = static_cast<unsigned int>(std::floor(indexes[0] * 0.5));
  unsigned int coldPos = static_cast<unsigned int>(std::floor(indexes[1] * 0.5));

  return {hotCandidates[hotPos], coldCandidates[coldPos]};
}

pair<Candidate, Candidate> getEndmembersASEBAL(float *ndvi, float *surface_temperature, float *albedo, float *net_radiation, float *soil_heat, int height_band, int width_band)
{
  vector<Candidate> hotCandidates;
  vector<Candidate> coldCandidates;

  vector<float> tsQuartile(3);
  vector<float> ndviQuartile(3);
  vector<float> albedoQuartile(3);
  get_quartiles(ndvi, ndviQuartile.data(), height_band, width_band, 0.25, 0.75, 0.75);
  get_quartiles(albedo, albedoQuartile.data(), height_band, width_band, 0.25, 0.50, 0.75);
  get_quartiles(surface_temperature, tsQuartile.data(), height_band, width_band, 0.25, 0.75, 0.75);

  float *ho = (float *)malloc(sizeof(float) * height_band * width_band);
  for (int i = 0; i < height_band * width_band; i++)
  {
    ho[i] = net_radiation[i] - soil_heat[i];

    bool hotAlbedo = !std::isnan(albedo[i]) && albedo[i] > albedoQuartile[1];
    bool hotNDVI = !std::isnan(ndvi[i]) && ndvi[i] > 0.10 && ndvi[i] < ndviQuartile[0];
    bool hotTS = !std::isnan(surface_temperature[i]) && surface_temperature[i] > tsQuartile[1];

    bool coldAlbedo = !std::isnan(albedo[i]) && albedo[i] < albedoQuartile[1];
    bool coldNDVI = !std::isnan(ndvi[i]) && ndvi[i] >= ndviQuartile[1];
    bool coldTS = !std::isnan(surface_temperature[i]) && surface_temperature[i] < tsQuartile[0];

    int line = i / width_band;
    int col = i % width_band;

    if (hotAlbedo && hotNDVI && hotTS)
      hotCandidates.emplace_back(ndvi[i], surface_temperature[i], net_radiation[i], soil_heat[i], ho[i], line, col);

    if (coldNDVI && coldAlbedo && coldTS)
      coldCandidates.emplace_back(ndvi[i], surface_temperature[i], net_radiation[i], soil_heat[i], ho[i], line, col);
  }

  if (hotCandidates.empty() || coldCandidates.empty())
  {
    cerr << "Pixel problem! - There are no final candidates";
    exit(15);
  }

  std::sort(hotCandidates.begin(), hotCandidates.end(), compare_candidate_temperature);
  std::sort(coldCandidates.begin(), coldCandidates.end(), compare_candidate_temperature);

  unsigned int hotPos = static_cast<unsigned int>(std::floor(hotCandidates.size() * 0.5));
  unsigned int coldPos = static_cast<unsigned int>(std::floor(coldCandidates.size() * 0.5));

  return {hotCandidates[hotPos], coldCandidates[coldPos]};
}
