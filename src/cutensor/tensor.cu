#include "tensor.h"

Tensor::Tensor()
{
  HANDLE_CUTENSOR_ERROR(cutensorCreate(&this->handle));
  HANDLE_ERROR(cudaStreamCreate(&this->stream));
}

void Tensor::createPlanWork(cutensorOperationDescriptor_t desc)
{
  // Optional (but recommended): ensure that the scalar type is correct.
  cutensorDataType_t scalarType;
  HANDLE_CUTENSOR_ERROR(cutensorOperationDescriptorGetAttribute(handle, desc, CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE, (void *)&scalarType, sizeof(scalarType)));
  assert(scalarType == CUTENSOR_R_32F);

  // Set the algorithm to use
  const cutensorAlgo_t algo = CUTENSOR_ALGO_DEFAULT;
  cutensorPlanPreference_t planPrefContraction;
  HANDLE_CUTENSOR_ERROR(cutensorCreatePlanPreference(handle, &planPrefContraction, algo, CUTENSOR_JIT_MODE_NONE));

  // Query workspace estimate
  uint64_t workspaceSizeEstimate = 0;
  const cutensorWorksizePreference_t workspacePref = CUTENSOR_WORKSPACE_DEFAULT;
  HANDLE_CUTENSOR_ERROR(cutensorEstimateWorkspaceSize(handle, desc, planPrefContraction, workspacePref, &workspaceSizeEstimate));

  // Create Contraction Plan
  HANDLE_CUTENSOR_ERROR(cutensorCreatePlan(handle, &plan, desc, planPrefContraction, workspaceSizeEstimate));

  // Optional: query actually used workspace
  HANDLE_CUTENSOR_ERROR(cutensorPlanGetAttribute(handle, plan, CUTENSOR_PLAN_REQUIRED_WORKSPACE, &actualWorkspaceSize, sizeof(actualWorkspaceSize)));
  assert(actualWorkspaceSize <= workspaceSizeEstimate);

  // Define the workspace
  if (actualWorkspaceSize > 0)
  {
    HANDLE_ERROR(cudaMalloc(&work, actualWorkspaceSize));
    assert(uintptr_t(work) % 128 == 0); // workspace must be aligned to 128 byte-boundary
  }
}

void Tensor::createNormalContraction(int height_band, int width_band)
{
  int dim_num = 2;
  std::vector<int> axis{'m', 'n'};
  std::vector<int64_t> axis_dim = {height_band, width_band};

  const uint32_t kAlignment = 128;

  // Define descriptors
  cutensorTensorDescriptor_t descA;
  cutensorTensorDescriptor_t descB;
  cutensorTensorDescriptor_t descC;
  HANDLE_CUTENSOR_ERROR(cutensorCreateTensorDescriptor(this->handle, &descA, dim_num, axis_dim.data(), NULL, CUTENSOR_R_32F, kAlignment));
  HANDLE_CUTENSOR_ERROR(cutensorCreateTensorDescriptor(this->handle, &descB, dim_num, axis_dim.data(), NULL, CUTENSOR_R_32F, kAlignment));
  HANDLE_CUTENSOR_ERROR(cutensorCreateTensorDescriptor(this->handle, &descC, dim_num, axis_dim.data(), NULL, CUTENSOR_R_32F, kAlignment));

  cutensorOperationDescriptor_t descContraction;
  const cutensorComputeDescriptor_t descCompute = CUTENSOR_COMPUTE_DESC_32F;
  HANDLE_CUTENSOR_ERROR(cutensorCreateContraction(this->handle,
                                                  &descContraction,
                                                  descA, axis.data(), /* unary operator A*/ CUTENSOR_OP_IDENTITY,
                                                  descB, axis.data(), /* unary operator B*/ CUTENSOR_OP_IDENTITY,
                                                  descC, axis.data(), /* unary operator C*/ CUTENSOR_OP_IDENTITY,
                                                  descC, axis.data(),
                                                  descCompute));
  createPlanWork(descContraction);
}
