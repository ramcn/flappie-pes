#include <stdio.h>
#include <stdlib.h>
#include "mkl.h"

int main(int argc, char *argv[])
{
    MKL_INT         m=1, n=1, k=2;
    MKL_INT         lda=2, ldb=1, ldc=1;
    float           alpha=1.0f, beta=0.0f;
    MKL_INT8        a[2];
    MKL_INT8        b[2];
    MKL_INT32       c[2];

    a[0] = 255;
    a[1] = 255;
    b[0] = 127;
    b[1] = 127;
    const MKL_INT8 ao = 0; const MKL_INT8 bo = 0; MKL_INT32 co = 0; CBLAS_OFFSET    offsetc=CblasFixOffset;
    cblas_gemm_s8u8s32(CblasRowMajor, CblasNoTrans, CblasNoTrans, offsetc,
                      m, n, k, alpha, a, lda, ao, b, ldb, bo, beta, c, ldc, &co);

    printf("\nC: %d\n", c[0]);

    return 0;
}
