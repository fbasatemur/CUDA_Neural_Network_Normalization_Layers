#pragma once
#include "CpuGpuMat.h"

void batchNormalize(struct CpuGpuMat* cgMat, float gama, float beta, float epsilon);