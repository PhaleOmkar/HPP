// Headers

#include <stdio.h>
#include <cuda.h>

#include <math.h>

#include "helper_timer.h"

// Global Variables
float *hostInput1 = NULL;
float *hostInput2 = NULL;
float *hostOutput = NULL;
float *gold = NULL;

float *deviceInput1 = NULL;
float *deviceInput2 = NULL;
float *deviceOutput = NULL;

float timeOnCPU;
float timeOnGPU;

// Kernel
__global__ void VecAdd(float *in1, float *in2, float *out, int len)
{
	// calculate the current thread index 
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	// calculate, if the thread is within the range of input
	if (i < len)
	{
		out[i] = in1[i] + in2[i];
	}

}

// main function
int main(int argc, char **argv)
{
	// function declarations
	void vecAddCPU(float *, float *, float *, int);
	void randomInit(float *, int);
	void cleanup(void);

	// variables
	int iArraySize = 11444777;

	int size = sizeof(float) * iArraySize;

	// allocate memory on host
	hostInput1 = (float *)malloc(size);
	if (!hostInput1)
	{
		printf("Out Of Memory on Host!\nTerminating...\n\n");
		cleanup();
		exit(EXIT_FAILURE);
	}

	hostInput2 = (float *)malloc(size);
	if (!hostInput2)
	{
		printf("Out Of Memory on Host!\nTerminating...\n\n");
		cleanup();
		exit(EXIT_FAILURE);
	}

	hostOutput = (float *)malloc(size);
	if (!hostOutput)
	{
		printf("Out Of Memory on Host!\nTerminating...\n\n");
		cleanup();
		exit(EXIT_FAILURE);
	}

	gold = (float *)malloc(size);
	if (!gold)
	{
		printf("Out Of Memory on Host!\nTerminating...\n\n");
		cleanup();
		exit(EXIT_FAILURE);
	}

	// allcate memory on device
	cudaError_t cuda_error = cudaSuccess;
	cuda_error = cudaMalloc((void**)&deviceInput1, size);
	if (cuda_error != cudaSuccess)
	{
		printf("Cannot Allocate Memory on Device!\nError: %s\nFile Name : %s, Line No: %d\n\n", cudaGetErrorString(cuda_error), __FILE__, __LINE__);
		cleanup();
		exit(EXIT_FAILURE);
	}

	cuda_error = cudaMalloc((void**)&deviceInput2, size);
	if (cuda_error != cudaSuccess)
	{
		printf("Cannot Allocate Memory on Device!\nError: %s\nFile Name : %s, Line No: %d\n\n", cudaGetErrorString(cuda_error), __FILE__, __LINE__);
		cleanup();
		exit(EXIT_FAILURE);
	}

	cuda_error = cudaMalloc((void**)&deviceOutput, size);
	if (cuda_error != cudaSuccess)
	{
		printf("Cannot Allocate Memory on Device!\nError: %s\nFile Name : %s, Line No: %d\n\n", cudaGetErrorString(cuda_error), __FILE__, __LINE__);
		cleanup();
		exit(EXIT_FAILURE);
	}


	// Initialize the input arrays!
	randomInit(hostInput1, iArraySize);
	randomInit(hostInput2, iArraySize);

	// timer
	StopWatchInterface *stopwatch = NULL;
	sdkCreateTimer(&stopwatch);

	// run on HOST!
	sdkStartTimer(&stopwatch);
	vecAddCPU(hostInput1, hostInput2, gold, iArraySize);
	sdkStopTimer(&stopwatch);
	timeOnCPU = sdkGetTimerValue(&stopwatch);
	sdkDeleteTimer(&stopwatch);


	// print the result!
	printf("Time on CPU: %f ms\n", timeOnCPU);
	
	
	// Copy data to Device!
	cuda_error = cudaMemcpy(deviceInput1, hostInput1, size, cudaMemcpyHostToDevice);
	if (cuda_error != cudaSuccess)
	{
		printf("Cannot Copy Memory From Host to Device!\nError: %s\nFile Name : %s, Line No: %d\n\n", cudaGetErrorString(cuda_error), __FILE__, __LINE__);
		cleanup();
		exit(EXIT_FAILURE);
	}

	cuda_error = cudaMemcpy(deviceInput2, hostInput2, size, cudaMemcpyHostToDevice);
	if (cuda_error != cudaSuccess)
	{
		printf("Cannot Copy Memory From Host to Device!\nError: %s\nFile Name : %s, Line No: %d\n\n", cudaGetErrorString(cuda_error), __FILE__, __LINE__);
		cleanup();
		exit(EXIT_FAILURE);
	}


	// Kernel Configuration
	dim3 GridDim = dim3(ceil(iArraySize / 256.0), 1, 1);
	dim3 BlockDim = dim3(256, 1, 1);

	// time the execution
	stopwatch = NULL;
	sdkCreateTimer(&stopwatch);

	// Let's run!
	sdkStartTimer(&stopwatch);
	VecAdd <<<GridDim, BlockDim >>> (deviceInput1, deviceInput2, deviceOutput, iArraySize);
	sdkStopTimer(&stopwatch);

	// Copy Result from Device Memory to Host Memory
	cuda_error = cudaMemcpy(hostOutput, deviceOutput, size, cudaMemcpyDeviceToHost);
	if (cuda_error != cudaSuccess)
	{
		printf("Cannot Copy Memory From Host to Device!\nError: %s\nFile Name : %s, Line No: %d\n\n", cudaGetErrorString(cuda_error), __FILE__, __LINE__);
		cleanup();
		exit(EXIT_FAILURE);
	}

	timeOnGPU = sdkGetTimerValue(&stopwatch);
	sdkDeleteTimer(&stopwatch);
	
	// print the result!
	printf("Time on GPU: %f ms\n", timeOnGPU);

	float epsilon = 0.000001f;
	bool bAccurate = true;
	int index = 0;
	
	for (int i = 0; i < iArraySize; i++)
	{
		if (fabs(gold[i] - hostOutput[i]) > epsilon)
		{
			bAccurate = false;
			index = i;
			break;
		}
	}

	if (bAccurate)
		printf("The Results are accurate upto %f %%\n\n", epsilon);
	else
		printf("The Results are not accurate, breaking index was: %d\n\n", index);

	return(0);
}


void randomInit(float *arr, int size)
{
	float fScale = 1.0 / (float)RAND_MAX;

	for (int i = 0; i < size; i++)
	{
		arr[i] = fScale * rand();
	}
}
void vecAddCPU(float *in1, float *in2, float *out, int size)
{
	for (int i = 0; i < size; i++)
	{
		out[i] = in1[i] + in2[i];
	}
}


void cleanup()
{
	if (deviceOutput)
	{
		cudaFree(deviceOutput);
		deviceOutput = NULL;
	}

	if (deviceInput2)
	{
		cudaFree(deviceInput2);
		deviceInput2 = NULL;
	}

	if (deviceInput1)
	{
		cudaFree(deviceInput1);
		deviceInput1 = NULL;
	}

	if (hostOutput)
	{
		free(hostOutput);
		hostOutput = NULL;
	}

	if (hostInput2)
	{
		free(hostInput2);
		hostInput2 = NULL;
	}

	if (hostInput1)
	{
		free(hostInput1);
		hostInput1 = NULL;
	}

}