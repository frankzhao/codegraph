/* CODEGRAPH GENERATED CODE BEGIN */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void codegraphKernel(float* a, float* c, const int chunkSize) {
    int threadid = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	// Don't calculate for elements outside of matrix
	if (threadid >= chunkSize)
		return;

    int chunkidx = threadid * chunkSize;
    
    // Calculate
    c[chunkidx] = a[chunkidx + 0]  *  a[chunkidx + 1]  +  a[chunkidx + 2]  *  a[chunkidx + 3]  +  a[chunkidx + 4];
     c[chunkidx] = a[chunkidx + 0]  +  a[chunkidx + 1]  *  a[chunkidx + 2]  +  a[chunkidx + 3]  *  a[chunkidx + 4];

}
int main() {
    const int chunkSize = 5;
    float initmem[10] = {
        (float) 2, (float) 3, (float) 2, (float) 1, (float) 0, (float) 0, (float) 3, (float) 2, (float) 3, (float) 4
    };


    // Copy to device
	float* dev_initmem = 0;
	float* dev_out = 0;
	float out[2] = {0.f, 0.f};
	cudaMalloc(&dev_initmem, initSize * sizeof(float));
	cudaMalloc(&dev_out, 2 * sizeof(float));
	cudaMemcpy(dev_initmem, initmem, initSize * sizeof(float), cudaMemcpyHostToDevice);
    // Run on device
    codegraphKernel<<<1,initSize>>>(dev_initmem, dev_out, chunkSize);
	// Copy results
	cudaMemcpy(out, dev_out, 2 * sizeof(float), cudaMemcpyDeviceToHost);

    /*
     *Do something with results here
     */

    // Free
 	cudaFree(dev_initmem);
 	cudaFree(dev_out);
}
/* CODEGRAPH GENERATED CODE END */
