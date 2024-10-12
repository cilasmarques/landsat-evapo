#include "tensor.cuh"

// This file is temporarily empty because the tensor implementation will be added in the cutensor folder.
// It is included here only to prevent compilation errors until the final code is completed.

Tensor::Tensor() {};

Tensor::Tensor(int height_band, int width_band) {}

void Tensor::createTrinary(cutensorPlan_t &plan, cutensorOperator_t OPA, cutensorOperator_t OPB, cutensorOperator_t OPC, cutensorOperator_t OPAB, cutensorOperator_t OPABC) {}

void Tensor::createBinary(cutensorPlan_t &plan, cutensorOperator_t OPA, cutensorOperator_t OPB, cutensorOperator_t OPAB) {}

void Tensor::createPermutation(cutensorPlan_t &plan, cutensorOperator_t OPA) {}

void Tensor::createPlan(cutensorPlan_t &plan, cutensorOperationDescriptor_t desc) {}
