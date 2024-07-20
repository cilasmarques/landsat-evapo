#include "cuda_utils.h"
#include "constants.h"

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
};
