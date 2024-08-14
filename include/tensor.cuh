#include "cuda_utils.h"
#include "constants.h"

#if __has_include(<cutensor.h>)
#include <cutensor.h>
#define CUTENSOR_AVAILABLE
#endif

#ifdef CUTENSOR_AVAILABLE
/**
 * This function checks the return value of the cuTENSOR call and exits
 * the application if the call failed.
 *
 * @param err: cuTENSOR error code.
 * @param file: File name where the error occurred.
 * @param line: Line number where the error occurred.
 */
static void HandleCuTensorError(cutensorStatus_t err, const char *file, int line)
{
    if (err != CUTENSOR_STATUS_SUCCESS)
    {
        printf("cuTENSOR error: %s in %s at line %d\n", cutensorGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

/**
 * Macro to handle cuTENSOR errors.
 */
#define HANDLE_CUTENSOR_ERROR(err) (HandleCuTensorError(err, __FILE__, __LINE__))
#endif

struct Tensor
{
    cutensorHandle_t handle;
    cudaStream_t stream;

    void *work;
    cutensorPlan_t plan;
    uint64_t actualWorkspaceSize;

    Tensor();

    void createPlanWork(cutensorOperationDescriptor_t desc);

    void createNormalContraction(int height_band, int width_band);

    void createAddMulTrinity(int height_band, int width_band);

    void createBinary(int height_band, int width_band, cutensorOperator_t OPA, cutensorOperator_t OPB, cutensorOperator_t OPAB);
};
