#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mkl.h>
#include <mkl_cblas.h>
#include <mkl_blas.h>


void PrintMatrix(float* pMatrix, const size_t nR, const size_t nC, const CBLAS_ORDER Order) {
    unsigned int i, j;
    if (Order == CblasRowMajor) {
        for (i = 0; i < nR; i++) {
            for (j = 0; j < nC; j++) {
                fprintf(stderr,"%f \t ", pMatrix[i * nC + j]); // !!!
            }
            fprintf(stderr,"\n"); // !!!
        }
    } else {
        for (i = 0; i < nR; i++) {
            for (j = 0; j < nC; j++) {
                fprintf(stderr,"%f \t ", pMatrix[i + j* nR ]); // !!!
            }
            fprintf(stderr,"\n"); // !!!
        }
    }
    fprintf(stderr,"\n"); // !!!
}

int main(void)
{
    const int m = 4;
    const int n = 8;
    const int k = 1;

    float A[] = {0.1,0.2,0.3,0.4,0.5,0.1,0.2,0.3,   0.1,0.2,0.3,0.4,0.5,0.1,0.2,0.3,   0.1,0.2,0.3,0.4,0.5,0.1,0.2,0.3,   0.1,0.2,0.3,0.4,0.5,0.1,0.2,0.1};
    float B[] = { 0.1, 0.2, 0.3, 0.4 };

    float alpha = 1.0, beta = 0.0;

    float *a1 = (float *) malloc(m * n *sizeof(float)); //4x8
    MKL_INT8 *a1_mkl_int8 = (MKL_INT8 *) malloc(m * n *sizeof(MKL_INT8)); //4x8
    float *b1 = (float *) malloc(m * 1 * sizeof(float)); // 4x1
    MKL_INT8 *b1_mkl_int8 = (MKL_INT8 *) malloc(m * 1 *sizeof(MKL_INT8)); //4x1
    float *c1 = (float *) malloc(n * 1 * sizeof(float)); // 8x1
    MKL_INT32 *c1_mkl_int32 = (MKL_INT32 *) malloc(n * 1 *sizeof(MKL_INT32)); //8x1
    memcpy(a1, A, m * n * sizeof(float));
    memcpy(b1, B, m * 1 * sizeof(float));
    memset(c1, 0, n * 1 * sizeof(float));
    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, n,k,m, alpha, a1, m, b1, n, beta, c1, n); // 8x4 multiplied to 4x1 gives 8x1
    fprintf(stderr,"Result after sgemm\n"); 
    PrintMatrix(c1, 8, k, CblasRowMajor); 

    for(int i=0;i<m;i++){
	for(int j=0;j<n;j++){
	   	MKL_INT8 x = (MKL_INT8) (*(a1+(i*m)+j) * 100); 
    		if(x > 127) x = 127;  
		if(x < -127) x = -127;
		*(a1_mkl_int8+(i*m)+j) = x;
	}
     }
    for(int j=0;j<m;j++){
	MKL_INT8 y = (MKL_INT8) (*(b1+j) * 100); 
    	if(y > 127) y = 127; 
	if(y < -127) y = -127; 
	*(b1_mkl_int8+j) = y;
    }

    const MKL_INT8 ao = 0; const MKL_INT8 bo = 0; MKL_INT32 co = 0; CBLAS_OFFSET offsetc=CblasFixOffset;
    cblas_gemm_s8u8s32(CblasColMajor, CblasTrans, CblasNoTrans, offsetc,
                      n, k, m, alpha, a1_mkl_int8, m, ao, b1_mkl_int8, n, bo, beta, c1_mkl_int32, n, &co);
    for(int j=0;j<n;j++) {
	 float y = (float) (*(c1_mkl_int32+j)); 
	 y = y / 10000;
	 *(c1+j) = y;
    }	    
	 
    fprintf(stderr,"Result after gemm_s8u8s32\n"); 
    PrintMatrix(c1, 8, k, CblasRowMajor); 

    free(a1); free(b1); free(c1); free(a1_mkl_int8);

    return 0;
}
