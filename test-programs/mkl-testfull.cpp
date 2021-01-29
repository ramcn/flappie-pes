    #include <iostream>
    using namespace std;
    #include <cmath>
    #include <cstdio>
    #include <cstdlib>
    #include <mkl.h>
    #include <mkl_cblas.h>
    #include <mkl_blas.h>
    #include <mkl_lapack.h>
    #include <mkl_lapacke.h>

    template<typename T>
    void printArray(T *data, char *name, int len){
        cout << name << "\n";
        for(int i=0;i<len;i++){
            cout << data[i] << " ";
        }
        cout << "\n";
    }

    template<typename T>
    void printMatrix(T *data, char *name, int m, int n){
        cout << name << "\n";
        for(int i=0;i<m;i++){
            for(int j=0;j<n;j++){
                cout << data[n*i+j] << " ";
            }
            cout << "\n";
        }
        cout << "\n";
    }

    int main()
    {
        int len=10;
        double *x=new double[len];
        double *y=new double[len];
        for(int i=0;i<len;i++){
            x[i]=(double)rand()/RAND_MAX;
            y[i]=(double)rand()/RAND_MAX;
        }
        printArray<double>(x, "x", len);
        printArray<double>(y, "y", len);

        //sum(x)
        double x_sum=cblas_dasum(len,x,1);
        cout<< "sum(x): "<< x_sum <<"\n";

        //y=a*x+y
        double alpha=1;
        cblas_daxpy(len,alpha,x,1,y,1);
        printArray<double>(y,"y=a*x+y",len);

        //y=x
        cblas_dcopy(len,x,1,y,1);
        printArray<double>(y,"y=x",len);

        //x*y';
        double xy_dot=cblas_ddot(len,x,1,y,1);
        cout <<"x*y': "<<xy_dot<<"\n";

        //extened x*y', not test
        //cblas_sdsdot(),not test
        //cbals_dsddot(),not test
        //for complex vector
        //cblas_dotc(),no this function
        //cblas_dotu(),no this function

        //norm(x), or ||x||_2
        double x_norm=cblas_dnrm2(len,x,1);
        cout<<"x_norm: "<<x_norm<<"\n";

        //x(i)=c*x(i)+s*y(i);
        //y(i)=c*y(i)-s*x(i);
        //cblas_zrotg(); not test

        //swap(x,y)
        cblas_dswap(len,x,1,y,1);
        printArray<double>(x,"x:",len);
        printArray<double>(y,"y:",len);


        //LEVEL 2 BLAS
        //matrix and vector manipulation
        int m=len;
        int n=len;
        double *A=new double[m*n];
        for(int i=0;i<m*n;i++){
            A[i]=(double)rand()/RAND_MAX;
        }
        printMatrix<double>(A,"A",m,n);

        //matrix and vector multiplication
        double alpha_dgemv=1.0;
        double beta_dgemv=1.0;
        //y=alpha*A*x+beta*y, if A is a mxn band matrix, then use cblas_dgbmv
        cblas_dgemv(CblasRowMajor,CblasNoTrans,m,n,alpha_dgemv,A,m,x,1,beta_dgemv,y,1);
        printArray<double>(x,"x:",len);
        printArray<double>(y,"y=alpha*A*x+beta*y",len);
        //y=alpha*A'*x+beta*y
        cblas_dgemv(CblasRowMajor,CblasTrans,m,n,alpha_dgemv,A,m,x,1,beta_dgemv,y,1);
        printArray<double>(x,"x:",len);
        printArray<double>(y,"y=alpha*A'*x+beta*y",len);

        //A=alpha*x*y'+A;
        double alpha_dger=1.0;
        cblas_dger(CblasRowMajor,m,n,alpha_dger,x,1,y,1,A,m);
        printArray<double>(x,"x:",len);
        printArray<double>(y,"y:",len);
        printMatrix<double>(A,"A=alpha1*x*y'+A",m,n);
        delete[] x;x=NULL;
        delete[] y;y=NULL;
        delete[] A;A=NULL;


        //
        m=10;
        n=5;
        int k=3;
        double *Amxk=new double[m*k];
        double *Bkxn=new double[k*n];
        double *Cmxn=new double[m*n];
        for(int i=0;i<m*k;i++){
            Amxk[i]=(double)rand()/RAND_MAX;
        }
        for(int i=0;i<k*n;i++){
            Bkxn[i]=(double)rand()/RAND_MAX;
        }
        for(int i=0;i<m*n;i++){
            Cmxn[i]=0;
        }
        printMatrix<double>(Amxk,"Amxk",m,k);
        printMatrix<double>(Bkxn,"Bkxn",k,n);
        printMatrix<double>(Cmxn,"Cmxn",m,n);
        double alpha_dgemm=1.0;
        double beta_dgemm=1.0;
        cblas_dgemm(CblasRowMajor,
                    CblasNoTrans,
                    CblasNoTrans,
                    m,
                    n,
                    k,
                    alpha_dgemm,
                    Amxk,
                    k,
                    Bkxn,
                    n,
                    beta_dgemm,
                    Cmxn,
                    n);
        printMatrix<double>(Cmxn,"Cmxn",m,n);
        delete[] Amxk;
        delete[] Bkxn;
        delete[] Cmxn;


        //general symmetric matrix eigenvalue decomposition
        /* Locals */
        const MKL_INT N=5;
        const MKL_INT LDA=5;
        MKL_INT  lda = LDA, info;
        n=N;
        /* Local arrays */
        double w[N];
        double a[LDA*N] = {
            6.39,  0.13, -8.23, 5.71, -3.18,
            0.00,  8.37, -4.46, -6.10,  7.21,
            0.00,  0.00, -9.58, -9.25, -7.42,
            0.00,  0.00, 0.00, 3.72,  8.54,
            0.00,  0.00, 0.00, 0.00,  2.51
        };
        /* Executable statements */
        printf( "LAPACKE_dsyevd (row-major, high-level) Example Program Results\n" );
        /* Solve eigenproblem */
        info = LAPACKE_dsyevd( LAPACK_ROW_MAJOR, 'V', 'U', n, a, lda, w );
        /* Check for convergence */
        if( info > 0 ) {
            printf( "The algorithm failed to compute eigenvalues.\n" );
            exit( 1 );
        }
        printArray<double>(w,"w:",N);
        printMatrix<double>(a,"a:",N,N);


        int N1=10;
        int d=3;
        double *A1=new double[N1*d];
        for(int j=0;j<d;j++){
        for(int i=0;i<N1;i++){

                A1[j*N1+i]=(double)rand()/RAND_MAX;
            }
        }
        printMatrix<double>(A1,"A:",d,N1);

        double *A_mean=new double[1*d];
        for(int i=0;i<d;i++){
            A_mean[i]=cblas_dasum(N,A1+i*N1,1);
        }
        printArray<double>(A_mean,"A_mean",d);
        delete[] A1;A1=NULL;
        delete[] A_mean;A_mean=NULL;
        return 0;
    }
