
/* CODEGRAPH GENERATED CODE BEGIN */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

int main() {
    float initmem[10] = {
        (float) 4, (float) 3, (float) 2, (float) 3, (float) 0, (float) 0, (float) 2, (float) 1, (float) 2, (float) 3
    };

    
    int threadid = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    float ykzlxx_10 = initmem[threadid + 0]  *  initmem[threadid + 1]  +  initmem[threadid + 2]  *  initmem[threadid + 3]  +  initmem[threadid + 4];
     float gooqax_00 = initmem[threadid + 0]  +  initmem[threadid + 1]  *  initmem[threadid + 2]  +  initmem[threadid + 3]  *  initmem[threadid + 4];
}
/* CODEGRAPH GENERATED CODE END */
