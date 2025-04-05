#include "cuda_utils.h"
#include "sensors.cuh"

MTL::MTL()
{
    this->year = 0;
    this->julian_day = 0;
    this->number_sensor = 0;
    this->sun_elevation = 0;
    this->rad_add = (float *)malloc(7 * sizeof(float));
    this->rad_mult = (float *)malloc(7 * sizeof(float));
    this->ref_add = (float *)malloc(7 * sizeof(float));
    this->ref_mult = (float *)malloc(7 * sizeof(float));
    this->ref_w_coeff = (float *)malloc(7 * sizeof(float));
};

MTL::MTL(string metadata_path)
{
    map<string, string> mtl;

    ifstream in(metadata_path);
    if (!in.is_open() || !in) {
        cerr << "Open metadata problem!" << endl;
        exit(2);
    }

    string line;
    while (getline(in, line)) {
        stringstream lineReader(line);
        string token;
        vector<string> nline;
        while (lineReader >> token)
            nline.push_back(token);

        if (nline.size() >= 3) {
            mtl[nline[0]] = nline[2];
        }
    }

    in.close();

    char julian_day[3];
    julian_day[0] = mtl["LANDSAT_SCENE_ID"][14];
    julian_day[1] = mtl["LANDSAT_SCENE_ID"][15];
    julian_day[2] = mtl["LANDSAT_SCENE_ID"][16];

    char year[4];
    year[0] = mtl["LANDSAT_SCENE_ID"][10];
    year[1] = mtl["LANDSAT_SCENE_ID"][11];
    year[2] = mtl["LANDSAT_SCENE_ID"][12];
    year[3] = mtl["LANDSAT_SCENE_ID"][13];

    int hours = atoi(mtl["SCENE_CENTER_TIME"].substr(1, 2).c_str());
    int minutes = atoi(mtl["SCENE_CENTER_TIME"].substr(4, 2).c_str());

    this->number_sensor = atoi(new char(mtl["LANDSAT_SCENE_ID"][3]));
    this->julian_day = atoi(julian_day);
    this->year = atoi(year);
    this->sun_elevation = atof(mtl["SUN_ELEVATION"].c_str());
    this->distance_earth_sun = atof(mtl["EARTH_SUN_DISTANCE"].c_str());
    this->image_hour = (hours + minutes / 60.0) * 100;

    this->rad_add = (float *)malloc(7 * sizeof(float));
    this->rad_mult = (float *)malloc(7 * sizeof(float));
    this->ref_add = (float *)malloc(7 * sizeof(float));
    this->ref_mult = (float *)malloc(7 * sizeof(float));
    this->ref_w_coeff = (float *)malloc(7 * sizeof(float));

    HANDLE_ERROR(cudaMalloc((void **)&this->rad_add_d, (7 * sizeof(float))));
    HANDLE_ERROR(cudaMalloc((void **)&this->rad_mult_d, (7 * sizeof(float))));
    HANDLE_ERROR(cudaMalloc((void **)&this->ref_add_d, (7 * sizeof(float))));
    HANDLE_ERROR(cudaMalloc((void **)&this->ref_mult_d, (7 * sizeof(float))));
    HANDLE_ERROR(cudaMalloc((void **)&this->ref_w_coeff_d, (7 * sizeof(float))));

    if (this->number_sensor == 8) {
        this->ref_w_coeff[PARAM_BAND_BLUE_INDEX] = 0.257048331f;
        this->ref_w_coeff[PARAM_BAND_GREEN_INDEX] = 0.251150748f;
        this->ref_w_coeff[PARAM_BAND_RED_INDEX] = 0.220943613f;
        this->ref_w_coeff[PARAM_BAND_NIR_INDEX] = 0.143411968f;
        this->ref_w_coeff[PARAM_BAND_SWIR1_INDEX] = 0.116657077f;
        this->ref_w_coeff[PARAM_BAND_TERMAL_INDEX] = 0.000000000f;
        this->ref_w_coeff[PARAM_BAND_SWIR2_INDEX] = 0.010788262f;

        this->rad_mult[PARAM_BAND_BLUE_INDEX] = atof(mtl["RADIANCE_MULT_BAND_2"].c_str());
        this->rad_mult[PARAM_BAND_GREEN_INDEX] = atof(mtl["RADIANCE_MULT_BAND_3"].c_str());
        this->rad_mult[PARAM_BAND_RED_INDEX] = atof(mtl["RADIANCE_MULT_BAND_4"].c_str());
        this->rad_mult[PARAM_BAND_NIR_INDEX] = atof(mtl["RADIANCE_MULT_BAND_5"].c_str());
        this->rad_mult[PARAM_BAND_SWIR1_INDEX] = atof(mtl["RADIANCE_MULT_BAND_6"].c_str());
        this->rad_mult[PARAM_BAND_TERMAL_INDEX] = atof(mtl["RADIANCE_MULT_BAND_10"].c_str());
        this->rad_mult[PARAM_BAND_SWIR2_INDEX] = atof(mtl["RADIANCE_MULT_BAND_7"].c_str());

        this->rad_add[PARAM_BAND_BLUE_INDEX] = atof(mtl["RADIANCE_ADD_BAND_2"].c_str());
        this->rad_add[PARAM_BAND_GREEN_INDEX] = atof(mtl["RADIANCE_ADD_BAND_3"].c_str());
        this->rad_add[PARAM_BAND_RED_INDEX] = atof(mtl["RADIANCE_ADD_BAND_4"].c_str());
        this->rad_add[PARAM_BAND_NIR_INDEX] = atof(mtl["RADIANCE_ADD_BAND_5"].c_str());
        this->rad_add[PARAM_BAND_SWIR1_INDEX] = atof(mtl["RADIANCE_ADD_BAND_6"].c_str());
        this->rad_add[PARAM_BAND_TERMAL_INDEX] = atof(mtl["RADIANCE_ADD_BAND_10"].c_str());
        this->rad_add[PARAM_BAND_SWIR2_INDEX] = atof(mtl["RADIANCE_ADD_BAND_7"].c_str());

        this->ref_mult[PARAM_BAND_BLUE_INDEX] = atof(mtl["REFLECTANCE_MULT_BAND_2"].c_str());
        this->ref_mult[PARAM_BAND_GREEN_INDEX] = atof(mtl["REFLECTANCE_MULT_BAND_3"].c_str());
        this->ref_mult[PARAM_BAND_RED_INDEX] = atof(mtl["REFLECTANCE_MULT_BAND_4"].c_str());
        this->ref_mult[PARAM_BAND_NIR_INDEX] = atof(mtl["REFLECTANCE_MULT_BAND_5"].c_str());
        this->ref_mult[PARAM_BAND_SWIR1_INDEX] = atof(mtl["REFLECTANCE_MULT_BAND_6"].c_str());
        this->ref_mult[PARAM_BAND_TERMAL_INDEX] = atof(mtl["REFLECTANCE_MULT_BAND_10"].c_str());
        this->ref_mult[PARAM_BAND_SWIR2_INDEX] = atof(mtl["REFLECTANCE_MULT_BAND_7"].c_str());

        this->ref_add[PARAM_BAND_BLUE_INDEX] = atof(mtl["REFLECTANCE_ADD_BAND_2"].c_str());
        this->ref_add[PARAM_BAND_GREEN_INDEX] = atof(mtl["REFLECTANCE_ADD_BAND_3"].c_str());
        this->ref_add[PARAM_BAND_RED_INDEX] = atof(mtl["REFLECTANCE_ADD_BAND_4"].c_str());
        this->ref_add[PARAM_BAND_NIR_INDEX] = atof(mtl["REFLECTANCE_ADD_BAND_5"].c_str());
        this->ref_add[PARAM_BAND_SWIR1_INDEX] = atof(mtl["REFLECTANCE_ADD_BAND_6"].c_str());
        this->ref_add[PARAM_BAND_TERMAL_INDEX] = atof(mtl["REFLECTANCE_ADD_BAND_10"].c_str());
        this->ref_add[PARAM_BAND_SWIR2_INDEX] = atof(mtl["REFLECTANCE_ADD_BAND_7"].c_str());
    } else {
        this->ref_w_coeff[PARAM_BAND_BLUE_INDEX] = 0.298220602f;
        this->ref_w_coeff[PARAM_BAND_GREEN_INDEX] = 0.270097933f;
        this->ref_w_coeff[PARAM_BAND_RED_INDEX] = 0.230996896f;
        this->ref_w_coeff[PARAM_BAND_NIR_INDEX] = 0.155050651f;
        this->ref_w_coeff[PARAM_BAND_SWIR1_INDEX] = 0.033085493f;
        this->ref_w_coeff[PARAM_BAND_TERMAL_INDEX] = 0.000000000f;
        this->ref_w_coeff[PARAM_BAND_SWIR2_INDEX] = 0.012548425f;

        this->rad_mult[PARAM_BAND_BLUE_INDEX] = atof(mtl["RADIANCE_MULT_BAND_1"].c_str());
        this->rad_mult[PARAM_BAND_GREEN_INDEX] = atof(mtl["RADIANCE_MULT_BAND_2"].c_str());
        this->rad_mult[PARAM_BAND_RED_INDEX] = atof(mtl["RADIANCE_MULT_BAND_3"].c_str());
        this->rad_mult[PARAM_BAND_NIR_INDEX] = atof(mtl["RADIANCE_MULT_BAND_4"].c_str());
        this->rad_mult[PARAM_BAND_SWIR1_INDEX] = atof(mtl["RADIANCE_MULT_BAND_5"].c_str());
        this->rad_mult[PARAM_BAND_TERMAL_INDEX] = atof(mtl["RADIANCE_MULT_BAND_6"].c_str());
        this->rad_mult[PARAM_BAND_SWIR2_INDEX] = atof(mtl["RADIANCE_MULT_BAND_7"].c_str());

        this->rad_add[PARAM_BAND_BLUE_INDEX] = atof(mtl["RADIANCE_ADD_BAND_1"].c_str());
        this->rad_add[PARAM_BAND_GREEN_INDEX] = atof(mtl["RADIANCE_ADD_BAND_2"].c_str());
        this->rad_add[PARAM_BAND_RED_INDEX] = atof(mtl["RADIANCE_ADD_BAND_3"].c_str());
        this->rad_add[PARAM_BAND_NIR_INDEX] = atof(mtl["RADIANCE_ADD_BAND_4"].c_str());
        this->rad_add[PARAM_BAND_SWIR1_INDEX] = atof(mtl["RADIANCE_ADD_BAND_5"].c_str());
        this->rad_add[PARAM_BAND_TERMAL_INDEX] = atof(mtl["RADIANCE_ADD_BAND_6"].c_str());
        this->rad_add[PARAM_BAND_SWIR2_INDEX] = atof(mtl["RADIANCE_ADD_BAND_7"].c_str());

        this->ref_mult[PARAM_BAND_BLUE_INDEX] = atof(mtl["REFLECTANCE_MULT_BAND_1"].c_str());
        this->ref_mult[PARAM_BAND_GREEN_INDEX] = atof(mtl["REFLECTANCE_MULT_BAND_2"].c_str());
        this->ref_mult[PARAM_BAND_RED_INDEX] = atof(mtl["REFLECTANCE_MULT_BAND_3"].c_str());
        this->ref_mult[PARAM_BAND_NIR_INDEX] = atof(mtl["REFLECTANCE_MULT_BAND_4"].c_str());
        this->ref_mult[PARAM_BAND_SWIR1_INDEX] = atof(mtl["REFLECTANCE_MULT_BAND_5"].c_str());
        this->ref_mult[PARAM_BAND_TERMAL_INDEX] = atof(mtl["REFLECTANCE_MULT_BAND_6"].c_str());
        this->ref_mult[PARAM_BAND_SWIR2_INDEX] = atof(mtl["REFLECTANCE_MULT_BAND_7"].c_str());

        this->ref_add[PARAM_BAND_BLUE_INDEX] = atof(mtl["REFLECTANCE_ADD_BAND_1"].c_str());
        this->ref_add[PARAM_BAND_GREEN_INDEX] = atof(mtl["REFLECTANCE_ADD_BAND_2"].c_str());
        this->ref_add[PARAM_BAND_RED_INDEX] = atof(mtl["REFLECTANCE_ADD_BAND_3"].c_str());
        this->ref_add[PARAM_BAND_NIR_INDEX] = atof(mtl["REFLECTANCE_ADD_BAND_4"].c_str());
        this->ref_add[PARAM_BAND_SWIR1_INDEX] = atof(mtl["REFLECTANCE_ADD_BAND_5"].c_str());
        this->ref_add[PARAM_BAND_TERMAL_INDEX] = atof(mtl["REFLECTANCE_ADD_BAND_6"].c_str());
        this->ref_add[PARAM_BAND_SWIR2_INDEX] = atof(mtl["REFLECTANCE_ADD_BAND_7"].c_str());
    }

    HANDLE_ERROR(cudaMemcpy(this->rad_add_d, this->rad_add, 7 * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(this->rad_mult_d, this->rad_mult, 7 * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(this->ref_add_d, this->ref_add, 7 * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(this->ref_mult_d, this->ref_mult, 7 * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(this->ref_w_coeff_d, this->ref_w_coeff, 7 * sizeof(float), cudaMemcpyHostToDevice));
};

Station::Station()
{
    this->temperature_image = 0;
};

Station::Station(string station_data_path, float image_hour)
{
    ifstream in(station_data_path);
    if (!in.is_open() || !in) {
        cerr << "Open station data problem!" << endl;
        exit(2);
    }

    string line;
    while (getline(in, line)) {
        istringstream lineReader(line);
        vector<string> nline;
        string token;
        while (getline(lineReader, token, ';'))
            nline.push_back(token);

        if (nline.size())
            this->info.push_back(nline);
    }

    in.close();

    if (this->info.size() < 1) {
        cerr << "Station data empty!" << endl;
        exit(12);
    }

    float diff = fabs(atof(this->info[0][2].c_str()) - image_hour);
    this->temperature_image = atof(this->info[0][6].c_str());
    this->v6 = atof(this->info[0][5].c_str());
    this->v7_max = atof(this->info[0][6].c_str());
    this->v7_min = atof(this->info[0][6].c_str());
    this->latitude = atof(this->info[0][3].c_str());
    this->longitude = atof(this->info[0][4].c_str());

    for (int i = 1; i < this->info.size(); i++) {
        v7_max = max(v7_max, atof(this->info[i][6].c_str()));
        v7_min = min(v7_min, atof(this->info[i][6].c_str()));

        if (fabs(atof(this->info[i][2].c_str()) - image_hour) < diff) {
            diff = fabs(atof(this->info[i][2].c_str()) - image_hour);
            this->temperature_image = atof(this->info[i][6].c_str());
            this->v6 = atof(this->info[i][5].c_str());
        }
    }
};

Landsat::Landsat()
{
    for (int i = 0; i < 9; i++)
        this->landsat_bands[i] = NULL;
};

Landsat::Landsat(string bands_paths[])
{
    // Open bands
    for (int i = 0; i < 8; i++) {
        std::string path_tiff_base = bands_paths[i];
        landsat_bands[i] = TIFFOpen(path_tiff_base.c_str(), "rm");
    }

    // Get bands metadata
    TIFFGetField(landsat_bands[1], TIFFTAG_IMAGELENGTH, &height_band);
    TIFFGetField(landsat_bands[1], TIFFTAG_IMAGEWIDTH, &width_band);
    TIFFGetField(landsat_bands[1], TIFFTAG_SAMPLEFORMAT, &sample_bands);
};

void saveTiff(string path, float *data, int height, int width)
{
    TIFF *tif = TIFFOpen(path.c_str(), "w");
    TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, width);
    TIFFSetField(tif, TIFFTAG_IMAGELENGTH, height);
    TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, 32);
    TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 1);
    TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, 1);
    TIFFSetField(tif, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
    TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
    TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP);
    TIFFSetField(tif, TIFFTAG_COMPRESSION, COMPRESSION_NONE);

    for (int i = 0; i < height; i++) {
        TIFFWriteScanline(tif, &data[i * width], i, 0);
    }

    TIFFClose(tif);
}

void printLinearPointer(float *pointer, int height, int width)
{
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            cout << pointer[i * width + j] << " ";
        }
        cout << endl;
    }
}
