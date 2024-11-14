#include "endmembers.h"
#include "filter.cuh"
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

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

string getEndmembersSTEEP(float *ndvi, float *d_ndvi, float *surface_temperature, float *d_surface_temperature, float *albedo, float *d_albedo,
                          float *net_radiation, float *d_net_radiation, float *soil_heat, float *d_soil_heat, int blocks_num, int threads_num,
                          Candidate &hot_pixel, Candidate &cold_pixel, int height_band, int width_band)
{
  string result = "";
  int64_t initial_time, final_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  Candidate *d_hotCandidates, *d_coldCandidates;
  const size_t MAXC = sizeof(Candidate) * height_band * width_band;

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

  float *d_ho;
  cudaMalloc((void **)&d_ho, sizeof(float) * height_band * width_band);

  int *d_indexes;
  int indexes[2] = {0, 0};
  cudaMalloc((void **)&d_indexes, sizeof(int) * 2);
  cudaMemcpy(d_indexes, indexes, sizeof(int) * 2, cudaMemcpyHostToDevice);

  Candidate *hotCandidates, *coldCandidates;
  hotCandidates = (Candidate *)malloc(MAXC);
  coldCandidates = (Candidate *)malloc(MAXC);

  vector<float> tsQuartile(3);
  vector<float> ndviQuartile(3);
  vector<float> albedoQuartile(3);

  try
  {
    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    cudaEventRecord(start);
    get_quartiles_cuda(d_ndvi, ndviQuartile.data(), height_band, width_band, 0.15, 0.97, 0.97, blocks_num, threads_num);
    get_quartiles_cuda(d_albedo, albedoQuartile.data(), height_band, width_band, 0.25, 0.50, 0.75, blocks_num, threads_num);
    get_quartiles_cuda(d_surface_temperature, tsQuartile.data(), height_band, width_band, 0.20, 0.85, 0.97, blocks_num, threads_num);

    process_pixels_STEEP<<<blocks_num, threads_num>>>(d_hotCandidates, d_coldCandidates, d_indexes,
                                                      d_ndvi, d_surface_temperature, d_albedo, d_net_radiation, d_soil_heat, d_ho,
                                                      ndviQuartile[0], ndviQuartile[1], tsQuartile[0], tsQuartile[1], tsQuartile[2],
                                                      albedoQuartile[0], albedoQuartile[1], albedoQuartile[2], height_band, width_band);
    cudaEventRecord(stop);

    cudaMemcpy(&indexes, d_indexes, sizeof(int) * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(hotCandidates, d_hotCandidates, sizeof(Candidate) * indexes[0], cudaMemcpyDeviceToHost);
    cudaMemcpy(coldCandidates, d_coldCandidates, sizeof(Candidate) * indexes[1], cudaMemcpyDeviceToHost);

    std::sort(hotCandidates, hotCandidates + indexes[0], compare_candidate_temperature);
    std::sort(coldCandidates, coldCandidates + indexes[1], compare_candidate_temperature);

    unsigned int hotPos = static_cast<unsigned int>(std::floor(indexes[0] * 0.5));
    unsigned int coldPos = static_cast<unsigned int>(std::floor(indexes[1] * 0.5));

    hot_pixel = hotCandidates[hotPos];
    cold_pixel = coldCandidates[coldPos];

    float cuda_time = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuda_time, start, stop);
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
    result += "KERNELS,PIXEL_FILTER," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
  }
  catch (const std::exception &e)
  {
    cerr << "Pixel filtering error: " << e.what() << endl;
    exit(15);
  }

  return result;
}

string getEndmembersASEBAL(float *ndvi, float *d_ndvi, float *surface_temperature, float *d_surface_temperature, float *albedo, float *d_albedo,
                           float *net_radiation, float *d_net_radiation, float *soil_heat, float *d_soil_heat, int blocks_num, int threads_num,
                           Candidate &hot_pixel, Candidate &cold_pixel, int height_band, int width_band)
{
  string result = "";
  int64_t initial_time, final_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  Candidate *d_hotCandidates, *d_coldCandidates;
  const size_t MAXC = sizeof(Candidate) * height_band * width_band;

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

  float *d_ho;
  cudaMalloc((void **)&d_ho, sizeof(float) * height_band * width_band);

  int *d_indexes;
  int indexes[2] = {0, 0};
  cudaMalloc((void **)&d_indexes, sizeof(int) * 2);
  cudaMemcpy(d_indexes, indexes, sizeof(int) * 2, cudaMemcpyHostToDevice);

  Candidate *hotCandidates, *coldCandidates;
  hotCandidates = (Candidate *)malloc(MAXC);
  coldCandidates = (Candidate *)malloc(MAXC);

  vector<float> tsQuartile(3);
  vector<float> ndviQuartile(3);
  vector<float> albedoQuartile(3);

  try
  {
    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    cudaEventRecord(start);
    get_quartiles_cuda(d_ndvi, ndviQuartile.data(), height_band, width_band, 0.25, 0.50, 0.75, blocks_num, threads_num);
    get_quartiles_cuda(d_albedo, albedoQuartile.data(), height_band, width_band, 0.25, 0.50, 0.75, blocks_num, threads_num);
    get_quartiles_cuda(d_surface_temperature, tsQuartile.data(), height_band, width_band, 0.25, 0.50, 0.75, blocks_num, threads_num);

    process_pixels_ASEBAL<<<blocks_num, threads_num>>>(d_hotCandidates, d_coldCandidates, d_indexes,
                                                       d_ndvi, d_surface_temperature, d_albedo, d_net_radiation, d_soil_heat, d_ho,
                                                       ndviQuartile[0], ndviQuartile[2], tsQuartile[1], tsQuartile[0],
                                                       albedoQuartile[1], albedoQuartile[1], height_band, width_band);
    cudaEventRecord(stop);

    cudaMemcpy(&indexes, d_indexes, sizeof(int) * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(hotCandidates, d_hotCandidates, sizeof(Candidate) * indexes[0], cudaMemcpyDeviceToHost);
    cudaMemcpy(coldCandidates, d_coldCandidates, sizeof(Candidate) * indexes[1], cudaMemcpyDeviceToHost);

    std::sort(hotCandidates, hotCandidates + indexes[0], compare_candidate_temperature);
    std::sort(coldCandidates, coldCandidates + indexes[1], compare_candidate_temperature);

    unsigned int hotPos = static_cast<unsigned int>(std::floor(indexes[0] * 0.5));
    unsigned int coldPos = static_cast<unsigned int>(std::floor(indexes[1] * 0.5));

    hot_pixel = hotCandidates[hotPos];
    cold_pixel = coldCandidates[coldPos];

    float cuda_time = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuda_time, start, stop);
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
    result += "KERNELS,PIXEL_FILTER," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
  }
  catch (const std::exception &e)
  {
    cerr << "Pixel filtering error: " << e.what() << endl;
    exit(15);
  }

  return result;
}
