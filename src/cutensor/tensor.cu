#include "tensor.cuh"

Tensor::Tensor() {};

Tensor::Tensor(int height_band, int width_band)
{
  HANDLE_CUTENSOR_ERROR(cutensorCreate(&this->handle));
  HANDLE_ERROR(cudaStreamCreate(&this->stream));
  this->dim_num = 2;
  this->axis = {'m', 'n'};
  this->axis_dim = {height_band, width_band};

  const uint32_t kAlignment = 128;

  // Define descriptors
  HANDLE_CUTENSOR_ERROR(cutensorCreateTensorDescriptor(this->handle, &this->descA, this->dim_num, this->axis_dim.data(), NULL, CUTENSOR_R_32F, kAlignment));
  HANDLE_CUTENSOR_ERROR(cutensorCreateTensorDescriptor(this->handle, &this->descB, this->dim_num, this->axis_dim.data(), NULL, CUTENSOR_R_32F, kAlignment));
  HANDLE_CUTENSOR_ERROR(cutensorCreateTensorDescriptor(this->handle, &this->descC, this->dim_num, this->axis_dim.data(), NULL, CUTENSOR_R_32F, kAlignment));
  HANDLE_CUTENSOR_ERROR(cutensorCreateTensorDescriptor(this->handle, &this->descD, this->dim_num, this->axis_dim.data(), NULL, CUTENSOR_R_32F, kAlignment));

  // Create tensors
  // == Trinary
  createTrinary(this->tensor_plan_trinity_add_mult, CUTENSOR_OP_IDENTITY, CUTENSOR_OP_IDENTITY, CUTENSOR_OP_IDENTITY, CUTENSOR_OP_ADD, CUTENSOR_OP_MUL);
  createTrinary(this->tensor_plan_trinity_mult_add, CUTENSOR_OP_IDENTITY, CUTENSOR_OP_IDENTITY, CUTENSOR_OP_IDENTITY, CUTENSOR_OP_MUL, CUTENSOR_OP_ADD);

  // == Binary
  createBinary(this->tensor_plan_binary_max, CUTENSOR_OP_IDENTITY, CUTENSOR_OP_IDENTITY, CUTENSOR_OP_MAX);
  createBinary(this->tensor_plan_binary_min, CUTENSOR_OP_IDENTITY, CUTENSOR_OP_IDENTITY, CUTENSOR_OP_MIN);
  createBinary(this->tensor_plan_binary_rcp, CUTENSOR_OP_RCP, CUTENSOR_OP_RCP, CUTENSOR_OP_MUL);
  createBinary(this->tensor_plan_binary_add, CUTENSOR_OP_IDENTITY, CUTENSOR_OP_IDENTITY, CUTENSOR_OP_ADD);
  createBinary(this->tensor_plan_binary_mult, CUTENSOR_OP_IDENTITY, CUTENSOR_OP_IDENTITY, CUTENSOR_OP_MUL);
  createBinary(this->tensor_plan_binary_div, CUTENSOR_OP_IDENTITY, CUTENSOR_OP_RCP, CUTENSOR_OP_MUL);
  createBinary(this->tensor_plan_binary_sqtr_add, CUTENSOR_OP_IDENTITY, CUTENSOR_OP_SQRT, CUTENSOR_OP_ADD);
  createBinary(this->tensor_plan_binary_log_mul, CUTENSOR_OP_IDENTITY, CUTENSOR_OP_LOG, CUTENSOR_OP_MUL);
  createBinary(this->tensor_plan_binary_exp_mul, CUTENSOR_OP_IDENTITY, CUTENSOR_OP_EXP, CUTENSOR_OP_MUL);

  // == Permutation
  createPermutation(this->tensor_plan_permute_id, CUTENSOR_OP_IDENTITY);
  createPermutation(this->tensor_plan_permute_exp, CUTENSOR_OP_EXP);
  createPermutation(this->tensor_plan_permute_log, CUTENSOR_OP_LOG);
}

void Tensor::createTrinary(cutensorPlan_t &plan, cutensorOperator_t OPA, cutensorOperator_t OPB, cutensorOperator_t OPC, cutensorOperator_t OPAB, cutensorOperator_t OPABC)
{
  cutensorOperationDescriptor_t desc;
  const cutensorComputeDescriptor_t descCompute = CUTENSOR_COMPUTE_DESC_32F;
  HANDLE_CUTENSOR_ERROR(cutensorCreateElementwiseTrinary(this->handle, &desc,
                                                         this->descA, this->axis.data(), OPA,
                                                         this->descB, this->axis.data(), OPB,
                                                         this->descC, this->axis.data(), OPC,
                                                         this->descD, this->axis.data(),
                                                         OPAB, OPABC,
                                                         descCompute));
  createPlan(plan, desc);
}

void Tensor::createBinary(cutensorPlan_t &plan, cutensorOperator_t OPA, cutensorOperator_t OPB, cutensorOperator_t OPAB)
{
  cutensorOperationDescriptor_t desc;
  const cutensorComputeDescriptor_t descCompute = CUTENSOR_COMPUTE_DESC_32F;
  HANDLE_CUTENSOR_ERROR(cutensorCreateElementwiseBinary(this->handle, &desc,
                                                        this->descA, this->axis.data(), OPA,
                                                        this->descB, this->axis.data(), OPB,
                                                        this->descC, this->axis.data(),
                                                        OPAB, descCompute));
  createPlan(plan, desc);
}

void Tensor::createPermutation(cutensorPlan_t &plan, cutensorOperator_t OPA)
{
  cutensorOperationDescriptor_t desc;
  const cutensorComputeDescriptor_t descCompute = CUTENSOR_COMPUTE_DESC_32F;
  HANDLE_CUTENSOR_ERROR(cutensorCreatePermutation(this->handle, &desc,
                                                  this->descA, this->axis.data(), OPA,
                                                  this->descB, this->axis.data(),
                                                  descCompute));
  createPlan(plan, desc);
}

void Tensor::createPlan(cutensorPlan_t &plan, cutensorOperationDescriptor_t desc)
{
  // Optional (but recommended): ensure that the scalar type is correct.
  cutensorDataType_t scalarType;
  HANDLE_CUTENSOR_ERROR(cutensorOperationDescriptorGetAttribute(this->handle, desc, CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE, (void *)&scalarType, sizeof(scalarType)));
  assert(scalarType == CUTENSOR_R_32F);

  // Set the algorithm to use
  const cutensorAlgo_t algo = CUTENSOR_ALGO_DEFAULT;
  cutensorPlanPreference_t planPref;
  HANDLE_CUTENSOR_ERROR(cutensorCreatePlanPreference(this->handle, &planPref, algo, CUTENSOR_JIT_MODE_NONE));

  // Query workspace estimate
  uint64_t workspaceSizeEstimate = 0;
  const cutensorWorksizePreference_t workspacePref = CUTENSOR_WORKSPACE_DEFAULT;
  HANDLE_CUTENSOR_ERROR(cutensorEstimateWorkspaceSize(this->handle, desc, planPref, workspacePref, &workspaceSizeEstimate));

  // Create Plan
  HANDLE_CUTENSOR_ERROR(cutensorCreatePlan(this->handle, &plan, desc, planPref, workspaceSizeEstimate));
}

