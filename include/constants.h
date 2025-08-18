#pragma once

#include <algorithm>
#include <assert.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <math.h>
#include <queue>
#include <set>
#include <sstream>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <thread>
#include <tiffio.h>
#include <time.h>
#include <vector>

// NAMESPACES
using namespace std;
using namespace std::chrono;

// GLOBAL VARIABLES DECLARATION
extern int model_method;

// CONSTANTS DECLARATION
#define PARAM_BAND_BLUE_INDEX 0
#define PARAM_BAND_GREEN_INDEX 1
#define PARAM_BAND_RED_INDEX 2
#define PARAM_BAND_NIR_INDEX 3
#define PARAM_BAND_SWIR1_INDEX 4
#define PARAM_BAND_TERMAL_INDEX 5
#define PARAM_BAND_SWIR2_INDEX 6

#define INPUT_BAND_ELEV_INDEX 8
#define INPUT_MTL_DATA_INDEX 9
#define INPUT_STATION_DATA_INDEX 10
#define OUTPUT_FOLDER 11
#define METHOD_INDEX 12

// Epsilon
const float EPS = 1e-7;

// Not a number
const float NaN = 0.0f/0.0f;

// Pi 
const float PI = 3.14159265358979323846f;

// Karman's constant
const float VON_KARMAN = 0.41f;

// Earth's gravity
const float GRAVITY = 9.81f;

// Atmospheric density
const float RHO = 1.15f;

// Specific heat of air
const float SPECIFIC_HEAT_AIR = 1004.0f;

// Solar constant
const float GSC = 0.082f;

// Agricultural field land cover value
// Available at https://mapbiomas.org/downloads_codigos
const int AGP = 14, PAS = 15, AGR = 18, CAP = 19, CSP = 20, MAP = 21;
