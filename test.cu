/* CODEGRAPH GENERATED CODE BEGIN */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

float initmem[10] = {
    (float) 3, (float) 3, (float) 4, (float) 3, (float) 1, (float) 0, (float) 0, (float) 2, (float) 2, (float) 2
};

int main() {
    float fjcnca10 = initmem[0];
    float wmvhrb10 = initmem[1];
    float bseqza11 = initmem[2];
    float zerkdb10 = initmem[3];
    float wpreka00 = initmem[4];
    float xfvhfx_init = initmem[5];
    float gbfujx_init = initmem[6];
    float gcccoa01 = initmem[7];
    float nqvdib00 = initmem[8];
    float ncpqwb00 = initmem[9];
    float oelmgx_10 = bseqza11  *  zerkdb10  +  gbfujx_init  +  fjcnca10  *  nqvdib00;
    float sjarxx_00 = gcccoa01  *  wmvhrb10  +  xfvhfx_init  +  ncpqwb00  *  wpreka00;

    printf("%f\n", oelmgx_10);
    printf("%f\n", sjarxx_00);

}
/* CODEGRAPH GENERATED CODE END */