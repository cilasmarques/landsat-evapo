#ifndef UTILS_CUH
#define UTILS_CUH

#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>
#include <tiffio.h> // Include the libtiff header for TIFF type

struct Task {
    int band_idx;
    int line_idx;
    tdata_t buffer;
    uint32_t width;
    unsigned short scanline_size;
};

#endif 