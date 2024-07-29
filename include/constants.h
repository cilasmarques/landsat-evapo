#pragma once

#include <tiffio.h>
#include <math.h>
#include <iostream>
#include <string.h>
#include <string>
#include <fstream>
#include <sstream>
#include <map>
#include <vector>
#include <algorithm>
#include <time.h>
#include <chrono>
#include <stdlib.h>
#include <queue>
#include <set>
#include <thread>
#include <assert.h>

using namespace std;
using namespace std::chrono;

// CONSTANTS DECLARATION

#define PARAM_BAND_BLUE_INDEX       0
#define PARAM_BAND_GREEN_INDEX      1
#define PARAM_BAND_RED_INDEX        2
#define PARAM_BAND_NIR_INDEX        3
#define PARAM_BAND_SWIR1_INDEX      4
#define PARAM_BAND_TERMAL_INDEX     5
#define PARAM_BAND_SWIR2_INDEX      6

// Epsilon
const float EPS = 1e-7;

// Not a number
const float NaN = -sqrt(-1.0);

// Pi
const float PI = acos(-1);

// Karman's constant
const float VON_KARMAN = 0.41;

// Earth's gravity
const float GRAVITY = 9.81;

// Atmospheric density
const float RHO = 1.15;

// Specific heat of air
const float SPECIFIC_HEAT_AIR = 1004;

// Solar constant
const float GSC = 0.082;

// Agricultural field land cover value
// Available at https://mapbiomas.org/downloads_codigos
const int AGP = 14, PAS = 15, AGR = 18, CAP = 19, CSP = 20, MAP = 21;
