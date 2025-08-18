#pragma once

#include <vector>
#include <thread>
#include <functional>

// Global variable for number of threads (will be set via command line)
extern int NUM_THREADS;

/**
 * @brief Executes a given function in parallel across multiple threads.
 *
 * Divides a range of work (e.g., rows of an image) among a fixed number of threads
 * and invokes the provided function for each sub-range.
 *
 * @param total_size The total size of the work to be divided (e.g., total number of rows).
 * @param work_function A function that takes two integer arguments (start_index, end_index)
 *                      representing the portion of work for a single thread.
 */
inline void parallel_for(int total_size, const std::function<void(int, int)>& work_function) {
    if (total_size == 0) {
        return;
    }
    std::vector<std::thread> threads;
    int chunk_size = (total_size + NUM_THREADS - 1) / NUM_THREADS; // Ceiling division

    for (int i = 0; i < NUM_THREADS; ++i) {
        int start_row = i * chunk_size;
        int end_row = std::min(total_size, (i + 1) * chunk_size);

        if (start_row < end_row) { // Ensure there's work to do
            threads.emplace_back(work_function, start_row, end_row);
        }
    }

    for (auto& t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }
}