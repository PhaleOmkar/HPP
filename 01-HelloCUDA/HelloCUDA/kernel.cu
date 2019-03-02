// Headers
#include <stdio.h>
#include <cuda.h>

// Global Variables
float *hostInput1 = NULL;
float *hostInput2 = NULL;
float *hostOutput = NULL;

float *deviceInput1 = NULL;
float *deviceInput2 = NULL;
float *deviceOutput = NULL;

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
	void cleanup(void);

	// variables
	int iArraySize = 5;

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
	hostInput1[0] = 101.1;
	hostInput1[1] = 102.1;
	hostInput1[2] = 103.1;
	hostInput1[3] = 104.1;
	hostInput1[4] = 105.1;

	hostInput2[0] = 201.1;
	hostInput2[1] = 202.1;
	hostInput2[2] = 203.1;
	hostInput2[3] = 204.1;
	hostInput2[4] = 205.1;


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

	// Let's run!
	VecAdd << <GridDim, BlockDim >> > (deviceInput1, deviceInput2, deviceOutput, iArraySize);

	// Copy Result from Device Memory to Host Memory
	cuda_error = cudaMemcpy(hostOutput, deviceOutput, size, cudaMemcpyDeviceToHost);
	if (cuda_error != cudaSuccess)
	{
		printf("Cannot Copy Memory From Host to Device!\nError: %s\nFile Name : %s, Line No: %d\n\n", cudaGetErrorString(cuda_error), __FILE__, __LINE__);
		cleanup();
		exit(EXIT_FAILURE);
	}

	// print the result!
	for (int i = 0; i < iArraySize; i++)
	{
		printf("%f + %f = %f \n", hostInput1[i], hostInput2[i], hostOutput[i]);
	}

	return(0);
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