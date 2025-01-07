#include "cuda_utils.h"
#include "constants.h"

#if __has_include(<cutensor.h>)
#include <cutensor.h>
#define CUTENSOR_AVAILABLE
#endif

#ifndef TENSOR_CUH
#define TENSOR_CUH
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

struct Tensor
{
    int dim_num = 2;
    std::vector<int> axis{'m', 'n'};
    std::vector<int64_t> axis_dim;

    cudaStream_t stream;
    cutensorHandle_t handle;
    cutensorTensorDescriptor_t descA;
    cutensorTensorDescriptor_t descB;
    cutensorTensorDescriptor_t descC;
    cutensorTensorDescriptor_t descD;

    cutensorPlan_t tensor_plan_trinity_add_mult;
    cutensorPlan_t tensor_plan_trinity_mult_add;
    cutensorPlan_t tensor_plan_binary_max;
    cutensorPlan_t tensor_plan_binary_min;
    cutensorPlan_t tensor_plan_binary_rcp;
    cutensorPlan_t tensor_plan_binary_add;
    cutensorPlan_t tensor_plan_binary_div;
    cutensorPlan_t tensor_plan_binary_mult;
    cutensorPlan_t tensor_plan_binary_sqtr_add;
    cutensorPlan_t tensor_plan_binary_log_mul;
    cutensorPlan_t tensor_plan_binary_exp_mul;
    cutensorPlan_t tensor_plan_permute_id;
    cutensorPlan_t tensor_plan_permute_exp;
    cutensorPlan_t tensor_plan_permute_log;
    cutensorPlan_t tensor_plan_permute_sqtr;

    Tensor();

    Tensor(int height_band, int width_band);

    /**
     * Perform this operation:
     * - X = fABC(fAB(alpha * OP(A), beta * OP(B)), gamma * OP(C))
     */
    void createTrinary(cutensorPlan_t &plan, cutensorOperator_t OPA, cutensorOperator_t OPB, cutensorOperator_t OPC, cutensorOperator_t OPAB, cutensorOperator_t OPABC);

    /**
     * Perform this operation:
     * - X = fAB(alpha * OP(A), beta * OP(B))
     */
    void createBinary(cutensorPlan_t &plan, cutensorOperator_t OPA, cutensorOperator_t OPB, cutensorOperator_t OPAB);

    /**
     * Perform this operation:
     * - X = alpha * OP(A)
     */
    void createPermutation(cutensorPlan_t &plan, cutensorOperator_t OPA);

    /**
     * Create a cuTENSOR plan.
     *
     * @param plan: The cuTENSOR plan.
     * @param desc: The operation descriptor.
     */
    void createPlan(cutensorPlan_t &plan, cutensorOperationDescriptor_t desc);
};
#endif 