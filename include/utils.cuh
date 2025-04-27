#ifndef UTILS_CUH
#define UTILS_CUH

#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>
#include <tiffio.h> // Include the libtiff header for TIFF type

struct Barrier {
    std::mutex mtx;
    std::condition_variable cv;
    int count;
    int initial_count;
    
    Barrier(int count) : count(count), initial_count(count) {}
    
    void wait() {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [this] { return count == 0; });
    }
    
    void decrease() {
        std::unique_lock<std::mutex> lock(mtx);
        count--;
        if (count == 0) {
            cv.notify_all();
        }
    }
    
    void reset() {
        std::unique_lock<std::mutex> lock(mtx);
        count = initial_count;
    }
};

struct Task {
    int band_idx;
    TIFF* tiff;
    float* band;
    int height;
    int width;
};

// Declare the barrier as extern
extern Barrier barrier;

#endif 