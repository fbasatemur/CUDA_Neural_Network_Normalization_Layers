#include "device_launch_parameters.h"
#include <math.h>
#include "KernelGPU.cuh"


__global__ void gpuNormAndShift(float* gpuP, int size, float mean, float variance, float gamma, float beta, float epsilon)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if (id < size) {
		gpuP[id] = gamma * ((gpuP[id] - mean) / sqrt(variance + epsilon)) + beta;
	}
}

void gpuNormalizeAndShift(struct CpuGpuMat* cgMat, float mean, float variance, float gamma, float beta, float epsilon)
{
	int threadsPerBlock = 32;
	int blocksPerGrid = ceil(double(cgMat->Size) / double(threadsPerBlock));

	gpuNormAndShift << < blocksPerGrid, threadsPerBlock >> > ((float*)cgMat->gpuP, cgMat->Size, mean, variance, gamma, beta, epsilon);
}