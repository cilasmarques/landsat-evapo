#include "constants.h"
#include "sensors.cuh"
#include "surfaceData.cuh"

int blocks_n;
int threads_n;
int model_method;

/**
 * @brief Main function
 * This function is responsible for reading the input parameters and calling the Landsat class to process the products.
 *
 * @param argc Number of input parameters
 * @param argv Input parameters
 *              - INPUT_BAND_BLUE_INDEX         = 1;
 *              - INPUT_BAND_GREEN_INDEX        = 2;
 *              - INPUT_BAND_RED_INDEX          = 3;
 *              - INPUT_BAND_NIR_INDEX          = 4;
 *              - INPUT_BAND_SWIR1_INDEX        = 5;
 *              - INPUT_BAND_TERMAL_INDEX       = 6;
 *              - INPUT_BAND_SWIR2_INDEX        = 7;
 *              - INPUT_BAND_ELEVATION_INDEX    = 8;
 *              - INPUT_MTL_DATA_INDEX          = 9;
 *              - INPUT_STATION_DATA_INDEX      = 10;
 *              - INPUT_LAND_COVER_INDEX        = 11;
 *              - OUTPUT_FOLDER                 = 12;
 * @return int
 */
int main(int argc, char *argv[])
{
    // Load the meteorologic stations data
    string path_meta_file = argv[INPUT_MTL_DATA_INDEX];
    string station_data_path = argv[INPUT_STATION_DATA_INDEX];

    // Load the landsat images bands
    string bands_paths[INPUT_BAND_ELEV_INDEX];
    for (int i = 0; i < INPUT_BAND_ELEV_INDEX; i++) {
        bands_paths[i] = argv[i + 1];
    }

    // Load the SEB model (SEBAL or STEEP)
    if (argc >= METHOD_INDEX) {
        string flag = argv[METHOD_INDEX];
        if (flag.substr(0, 6) == "-meth=")
            model_method = flag[6] - '0';
    }

    // Load the number of threads for each block
    if (argc >= THREADS_INDEX) {
        string threads_flag = argv[THREADS_INDEX];
        if (threads_flag.substr(0, 9) == "-threads=")
            threads_n = atof(threads_flag.substr(9, threads_flag.size()).c_str());
        else
            threads_n = 32;
    }

    // Save output paths
    string output_folder = argv[OUTPUT_FOLDER];
    string output_time = output_folder + "/time.csv";

    // Time output
    ofstream time_output;
    time_output.open(output_time);
    string general, initial, final;
    system_clock::time_point begin, end;
    int64_t initial_time, final_time;
    float general_time;

    // Instantiate classes
    MTL mtl = MTL(path_meta_file);
    Landsat landsat = Landsat(bands_paths);
    Station station = Station(station_data_path, mtl.image_hour);
    Products products = Products(landsat.width_band, landsat.height_band);
    blocks_n = (landsat.width_band * landsat.height_band + threads_n - 1) / threads_n;

    // ===== RUN ALGORITHM =====
    begin = system_clock::now();
    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
    time_output << "STRATEGY,PHASE,TIMESTAMP,START_TIME,END_TIME" << std::endl;

    // process products
    time_output << products.read_data(landsat.landsat_bands);
    time_output << products.compute_Rn_G(products, station, mtl);
    time_output << products.select_endmembers();
    time_output << products.converge_rah_cycle(products, station);
    time_output << products.compute_H_ET(products, station, mtl);

    // save products
    // time_output << products.host_data();
    // time_output << products.save_products(output_folder);
    // time_output << products.print_products(output_folder);
    // products.close(landsat.landsat_bands);

    end = system_clock::now();
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
    general_time = duration_cast<nanoseconds>(end - begin).count() / 1000000.0;
    time_output << "KERNELS,P_TOTAL," << general_time << "," << initial_time << "," << final_time << std::endl;

    time_output.close();

    return 0;
}
