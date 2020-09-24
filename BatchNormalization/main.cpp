#include "CpuGpuMat.h"
#include "KernelGpu.cuh"
#include <cuda_runtime_api.h>						// cudaDeviceSynchronize()
#include "BatchNormalization.h"
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>

int main()
{
	// matrix
	struct CpuGpuMat cgMat;
	cgMat.Rows = 4;
	cgMat.Cols = 4;

	// matrix size
	cgMat.Size = cgMat.Rows * cgMat.Cols;			


	// cpu and gpu memory allocation
	cgMat.cpuP = (void*)malloc(cgMat.Size * sizeof(float));

	cudaError_t result = cudaMalloc(&cgMat.gpuP, cgMat.Size * sizeof(float));
	assert(result == cudaSuccess);

				
	// set values to cpu memory
	float* cpuFloatP = (float*)cgMat.cpuP;

	for (int i = 0; i < cgMat.Size; i++)
		cpuFloatP[i] = (float)i;

	/*
		Host => ram
		Device => graphics memory
	*/

	// Host -> Device
	cudaError_t result1 = cudaMemcpy(cgMat.gpuP, cgMat.cpuP, cgMat.Size * sizeof(float), cudaMemcpyHostToDevice);
	assert(result1 == cudaSuccess);					


	/// batchNormalize layer ----------------------------------
	float gamma = 1.0F;
	float beta = 0.0F;
	float epsilon = 0.001F;

	batchNormalize(&cgMat, gamma, beta, epsilon);									
	/// -------------------------------------------------------


	// Device -> Host
	cudaError_t result2 = cudaMemcpy(cgMat.cpuP, cgMat.gpuP, cgMat.Size * sizeof(float), cudaMemcpyDeviceToHost);
	assert(result2 == cudaSuccess);


	// show result of batchNormalize
	cpuFloatP = (float*)cgMat.cpuP;

	for (int i = 0; i < cgMat.Size; i++)
		printf("%d \t %f \n", i, cpuFloatP[i]);									


	cudaDeviceSynchronize();
	
	// cpu and gpu memory free
	cudaError_t result3 = cudaFree(cgMat.gpuP);
	assert(result3 == cudaSuccess);

	free(cgMat.cpuP);

	return 0;
}