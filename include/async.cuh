#ifndef ASYNC_CUH
#define ASYNC_CUH

#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>
#include <tiffio.h> // Include the libtiff header for TIFF type

struct Task {
    int band_idx;
    int strip_idx;
    tdata_t buffer;
    uint32_t pixels_in_strip;
};

#endif 