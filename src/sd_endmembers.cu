#include "constants.h"
#include "kernels.cuh"
#include "surfaceData.cuh"

__host__ __device__ Endmember::Endmember()
{
    this->ndvi = 0;
    this->temperature = 0;
    this->line = 0;
    this->col = 0;
}

__host__ __device__ Endmember::Endmember(double ndvi, double temperature, int line, int col)
{
    this->ndvi = ndvi;
    this->temperature = temperature;
    this->line = line;
    this->col = col;
}

void get_quartiles_cuda(double *d_target, double *v_quartile, int height_band, int width_band, double first_interval, double middle_interval, double last_interval, int blocks_n, int threads_n)
{
    double *d_filtered;
    cudaMalloc(&d_filtered, sizeof(double) * height_band * width_band);

    int indexes[1] = {0};
    int *indexes_d;
    cudaMalloc((void **)&indexes_d, sizeof(int) * 1);
    cudaMemcpy(indexes_d, indexes, sizeof(int) * 1, cudaMemcpyHostToDevice);

    filter_valid_values<<<blocks_n, threads_n>>>(d_target, d_filtered, indexes_d);

    cudaMemcpy(&indexes[0], indexes_d, sizeof(int), cudaMemcpyDeviceToHost);

    // Use Thrust to sort the valid elements on the GPU
    thrust::device_ptr<double> d_filtered_ptr = thrust::device_pointer_cast(d_filtered);
    thrust::sort(thrust::device, d_filtered_ptr, d_filtered_ptr + indexes[0]);

    int first_index = static_cast<int>(floor(first_interval * indexes[0]));
    int middle_index = static_cast<int>(floor(middle_interval * indexes[0]));
    int last_index = static_cast<int>(floor(last_interval * indexes[0]));

    double temp_value;
    cudaMemcpy(&temp_value, d_filtered + first_index, sizeof(double), cudaMemcpyDeviceToHost);
    v_quartile[0] = (double)temp_value;
    cudaMemcpy(&temp_value, d_filtered + middle_index, sizeof(double), cudaMemcpyDeviceToHost);
    v_quartile[1] = (double)temp_value;
    cudaMemcpy(&temp_value, d_filtered + last_index, sizeof(double), cudaMemcpyDeviceToHost);
    v_quartile[2] = (double)temp_value;

    // Free GPU memory
    cudaFree(d_filtered);
    cudaFree(indexes_d);
}

string getEndmembersSTEEP(Products products, int *indexes_d)
{
    string result = "";
    int64_t initial_time, final_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    vector<double> tsQuartile(3);
    vector<double> ndviQuartile(3);
    vector<double> albedoQuartile(3);

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    cudaEventRecord(start);
    get_quartiles_cuda(products.ndvi_d, ndviQuartile.data(), products.height_band, products.width_band, 0.15, 0.97, 0.97, blocks_n, threads_n);
    get_quartiles_cuda(products.albedo_d, albedoQuartile.data(), products.height_band, products.width_band, 0.25, 0.50, 0.75, blocks_n, threads_n);
    get_quartiles_cuda(products.surface_temperature_d, tsQuartile.data(), products.height_band, products.width_band, 0.20, 0.85, 0.97, blocks_n, threads_n);

    process_pixels_STEEP<<<blocks_n, threads_n>>>(products.hotCandidates_d, products.coldCandidates_d, indexes_d, products.ndvi_d, products.surface_temperature_d, products.albedo_d, products.net_radiation_d, products.soil_heat_d, ndviQuartile[0], ndviQuartile[1], tsQuartile[0], tsQuartile[1], tsQuartile[2], albedoQuartile[0], albedoQuartile[1], albedoQuartile[2]);
    cudaEventRecord(stop);

    float cuda_time = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuda_time, start, stop);
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
    return "KERNELS,PIXEL_FILTER," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
}

string getEndmembersASEBAL(Products products, int *indexes_d)
{
    string result = "";
    int64_t initial_time, final_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    vector<double> tsQuartile(3);
    vector<double> ndviQuartile(3);
    vector<double> albedoQuartile(3);

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    cudaEventRecord(start);
    get_quartiles_cuda(products.ndvi_d, ndviQuartile.data(), products.height_band, products.width_band, 0.25, 0.50, 0.75, blocks_n, threads_n);
    get_quartiles_cuda(products.albedo_d, albedoQuartile.data(), products.height_band, products.width_band, 0.25, 0.50, 0.75, blocks_n, threads_n);
    get_quartiles_cuda(products.surface_temperature_d, tsQuartile.data(), products.height_band, products.width_band, 0.25, 0.50, 0.75, blocks_n, threads_n);

    process_pixels_ASEBAL<<<blocks_n, threads_n>>>(products.hotCandidates_d, products.coldCandidates_d, indexes_d, products.ndvi_d, products.surface_temperature_d, products.albedo_d, products.net_radiation_d, products.soil_heat_d, ndviQuartile[0], ndviQuartile[2], tsQuartile[2], tsQuartile[0], albedoQuartile[2], albedoQuartile[1]);
    cudaEventRecord(stop);

    float cuda_time = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuda_time, start, stop);
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
    return "KERNELS,PIXEL_FILTER," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
}

string Products::select_endmembers(Products products)
{
    string result = "";
    int64_t initial_time, final_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    try
    {
        initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
        cudaEventRecord(start);

        int *indexes_d;
        int indexes[2] = {0, 0};
        cudaMalloc((void **)&indexes_d, sizeof(int) * 2);
        cudaMemcpy(indexes_d, indexes, sizeof(int) * 2, cudaMemcpyHostToDevice);

        if (model_method == 0)
            result += getEndmembersSTEEP(products, indexes_d);
        else if (model_method == 1)
            result += getEndmembersASEBAL(products, indexes_d);

        cudaMemcpy(&indexes, indexes_d, sizeof(int) * 2, cudaMemcpyDeviceToHost);
        int hot_pos = static_cast<unsigned int>(std::floor(indexes[0] * 0.5));
        int cold_pos = static_cast<unsigned int>(std::floor(indexes[1] * 0.5));

        if (indexes[0] == 0)
            throw std::runtime_error("No hot candidates found");
        if (indexes[1] == 0)
            throw std::runtime_error("No cold candidates found");

        // The dev_ptr_hot sort also sorts the hotCandidates_d array
        thrust::device_ptr<Endmember> dev_ptr_hot(products.hotCandidates_d);
        thrust::sort(dev_ptr_hot, dev_ptr_hot + indexes[0], CompareEndmemberTemperature());

        // The dev_ptr_cold sort also sorts the coldCandidates_d array
        thrust::device_ptr<Endmember> dev_ptr_cold(products.coldCandidates_d);
        thrust::sort(dev_ptr_cold, dev_ptr_cold + indexes[1], CompareEndmemberTemperature());

        Endmember hotCandidate = Endmember();
        Endmember coldCandidate = Endmember();

        cudaMemcpy(&hotCandidate, products.hotCandidates_d + hot_pos, sizeof(Endmember), cudaMemcpyDeviceToHost);
        cudaMemcpy(&coldCandidate, products.coldCandidates_d + cold_pos, sizeof(Endmember), cudaMemcpyDeviceToHost);

        cudaMemcpyToSymbol(hotEndmemberLine_d, &hotCandidate.line, sizeof(int), 0, cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol(hotEndmemberCol_d, &hotCandidate.col, sizeof(int), 0, cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol(coldEndmemberLine_d, &coldCandidate.line, sizeof(int), 0, cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol(coldEndmemberCol_d, &coldCandidate.col, sizeof(int), 0, cudaMemcpyHostToDevice);
        cudaEventRecord(stop);

        float cuda_time = 0;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&cuda_time, start, stop);
        final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
        result += "KERNELS,P2_PIXEL_SEL," + std::to_string(cuda_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
        return result;
    }
    catch (const std::exception &e)
    {
        cerr << "Pixel filtering error: " << e.what() << endl;
        exit(15);
    }

    return result;
}
