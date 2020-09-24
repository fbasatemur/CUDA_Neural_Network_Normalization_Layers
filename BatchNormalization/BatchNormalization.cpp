#include "CpuGpuMat.h"
#include "KernelGpu.cuh"
#include <math.h>
#include "BatchNormalization.h"

float mean(float* cpuP, const int& size)
{
	float sum = 0.0;
	for (int i = 0; i < size; i++)
		sum += cpuP[i];

	return sum / size;
}

float variance(float* cpuP, const int& size, const float& mean)
{
	float sum = 0.0;

	for (int i = 0; i < size; i++)
		sum += pow((cpuP[i] - mean), 2);

	return sum / size;
}

void batchNormalize(struct CpuGpuMat* cgMat, float gamma, float beta, float epsilon)
{
	float mean1 = mean((float*)cgMat->cpuP, cgMat->Size);

	float variance1 = variance((float*)cgMat->cpuP, cgMat->Size, mean1);

	gpuNormalizeAndShift(cgMat, mean1, variance1, gamma, beta, epsilon);
}