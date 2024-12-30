#include <fstream>
#include <iostream>

#include "utils.h"
#include "landsat.h"
#include "constants.h"
#include "parameters.h"

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
  int INPUT_BAND_ELEV_INDEX = 8;
  int INPUT_MTL_DATA_INDEX = 9;
  int INPUT_STATION_DATA_INDEX = 10;
  int OUTPUT_FOLDER = 11;
  int METHOD_INDEX = 12;
  int THREADS_INDEX = 13;

  // Load the meteorologic stations data
  string path_meta_file = argv[INPUT_MTL_DATA_INDEX];
  string station_data_path = argv[INPUT_STATION_DATA_INDEX];

  // Load the landsat images bands
  string bands_paths[INPUT_BAND_ELEV_INDEX];
  for (int i = 0; i < INPUT_BAND_ELEV_INDEX; i++)
  {
    bands_paths[i] = argv[i + 1];
  }

  // Load the SEB model (SEBAL or STEEP)
  int method = 0;
  if (argc >= METHOD_INDEX)
  {
    string flag = argv[METHOD_INDEX];
    if (flag.substr(0, 6) == "-meth=")
      method = flag[6] - '0';
  }

  // Load the number of threads for each block
  int threads_num = 1024;
  if (argc >= THREADS_INDEX)
  {
    string threads_flag = argv[THREADS_INDEX];
    if (threads_flag.substr(0, 9) == "-threads=")
      threads_num = atof(threads_flag.substr(9, threads_flag.size()).c_str());
  }

  // Save output paths
  string output_folder = argv[OUTPUT_FOLDER];
  string output_time = output_folder + "/time.csv";

  // time output
  ofstream time_output;
  time_output.open(output_time);
  string general, initial, final;
  system_clock::time_point begin, end, read_begin, read_end;
  int64_t initial_time, final_time, read_initial_time, read_final_time;
  float general_time, read_general_time;

  // ===== RUN ALGORITHM =====
  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
  time_output << "STRATEGY,PHASE,TIMESTAMP,START_TIME,END_TIME" << std::endl;

  // read input data
  read_begin = system_clock::now();
  read_initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  MTL mtl = MTL(path_meta_file);
  Station station = Station(station_data_path, mtl.image_hour);
  Landsat landsat = Landsat(bands_paths, mtl, threads_num);

  read_end = system_clock::now();
  read_final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
  read_general_time = duration_cast<nanoseconds>(read_end - read_begin).count() / 1000000.0;
  time_output << "SERIAL,P0_READ_INPUT," << read_general_time << "," << read_initial_time << "," << read_final_time << std::endl;

  // process products
  time_output << landsat.compute_Rn_G(station);
  time_output << landsat.select_endmembers(method);
  time_output << landsat.converge_rah_cycle(station, method);
  time_output << landsat.compute_H_ET(station);

  // save products
  // time_output << landsat.copy_to_host();
  // time_output << landsat.save_products(output_folder);
  // time_output << landsat.print_products(output_folder);

  end = system_clock::now();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
  general_time = duration_cast<nanoseconds>(end - begin).count() / 1000000.0;
  time_output << "KERNELS,P_TOTAL," << general_time << "," << initial_time << "," << final_time << std::endl;

  time_output.close();
  landsat.close();

  return 0;
}
