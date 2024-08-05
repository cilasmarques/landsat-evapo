#pragma once

#include <stdio.h>
#include <stdlib.h>

#ifdef __has_include
  #if __has_include(<cuda.h>)
    #include <cuda.h>
    #define CUDA_AVAILABLE
  #endif

  #if __has_include(<cuda_runtime_api.h>)
    #include <cuda_runtime_api.h>
    #define CUDA_RUNTIME_API_AVAILABLE
  #endif

  #if __has_include(<cuda_profiler_api.h>)
    #include <cuda_profiler_api.h>
    #define CUDA_PROFILER_API_AVAILABLE
  #endif
#endif

#ifdef CUDA_RUNTIME_API_AVAILABLE
/**
 * This function checks the return value of the CUDA runtime call and exits
 * the application if the call failed.
 *
 * @param err: CUDA error code.
 * @param file: File name where the error occurred.
 * @param line: Line number where the error occurred.
 */
static void HandleError(cudaError_t err, const char *file, int line)
{
  if (err != cudaSuccess)
  {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}

/**
 * This macro checks the return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))
#endif

#ifdef CUDA_AVAILABLE
/**
 * This macro checks the return value of a CUDA driver call and exits
 * the application if the call failed.
 *
 * See cuda.h for error code descriptions.
 */
#define CHECK_CUDA_RESULT(N)                                       \
  {                                                                \
    CUresult result = N;                                           \
    if (result != 0)                                               \
    {                                                              \
      printf("CUDA call on line %d returned error %d\n", __LINE__, \
             result);                                              \
      exit(1);                                                     \
    }                                                              \
  }
#endif
