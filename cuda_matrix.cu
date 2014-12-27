
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__global__ void cudaMatrixMul(float** a, float** b, float** c) {
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int col = blockIdx.x*blockDim.x + threadIdx.x;

	// Don't calculate for elements outside of matrix
	if (row + col > 2 * sizeof(a)/sizeof(*a))
		return;

	float* arow = a[row];
	float* bcol = b[col];

	// Calculate
	int size = sizeof(arow)/sizeof(*arow);
	float total = 0.f;
	for (int i=0; i<size, i++;)
		total += arow[i] + bcol[i];
	
	c[row][col] = total;
}

__global__ void codegraphMatrixMul(float* a, float* c, const int chunkSize) {
	int threadid = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	// Don't calculate for elements outside of matrix
	if (threadid >= chunkSize)
		return;

	int chunkidx = threadid * chunkSize;

	// Calculate
	c[threadid] = a[chunkidx + 0]  *  a[chunkidx + 1]  +  a[chunkidx + 2]  *  a[chunkidx + 3]  +  a[chunkidx + 4];

}

int main()
{
	// Inital memory block
	const int initSize = 10;
	// Chunk size
	const int chunkSize = 5;
	const float initmem[initSize] = {
        (float) 4, (float) 3, (float) 2, (float) 3, (float) 0,
		(float) 0, (float) 2, (float) 1, (float) 2, (float) 3
    };

	// Copy to device
	float* dev_initmem = 0;
	float* dev_out = 0;
	float out[2] = {0.f, 0.f};
	cudaMalloc(&dev_initmem, initSize * sizeof(float));
	cudaMalloc(&dev_out, 2 * sizeof(float));
	cudaMemcpy(dev_initmem, initmem, initSize * sizeof(float), cudaMemcpyHostToDevice);

	// Run on device
	codegraphMatrixMul<<<1,initSize>>>(dev_initmem, dev_out, chunkSize);

	// Check results
	cudaMemcpy(out, dev_out, 2 * sizeof(float), cudaMemcpyDeviceToHost);
	printf("%f, %f\n", out[0], out[1]);

	// Free
	cudaFree(dev_initmem);
	cudaFree(dev_out);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaError_t cudaStatus;
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}
