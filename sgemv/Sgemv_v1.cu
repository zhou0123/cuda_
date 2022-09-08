#include <stdio.h>
#include <stdlib.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h> 
#define FETCH_FLOAT4(pointer)(reinterpret_cast<float4*>(&(pointer))[0])
template <unsigned int WarpSize>
__device__ float warp_reduce(float sum)
{
    if (warpSize>=32) sum += __shfl_down_sync(0xffffffff, sum, 16); 
    if (warpSize>=16) sum += __shfl_down_sync(0xffffffff, sum, 8); 
    if (warpSize>=8) sum += __shfl_down_sync(0xffffffff, sum, 4); 
    if (warpSize>=4) sum += __shfl_down_sync(0xffffffff, sum, 2); 
    if (warpSize>=2) sum += __shfl_down_sync(0xffffffff, sum, 1); 
    return sum;
}
template<const int warpsize>
__global__ void Sgemv(float * mat , float * vector , float * o_vector,int M, int N)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;

    const int col = tx%warpsize;
    const int row = ty+blockDim.y*bx;

    if (row < M)
    {
        int nitor = (N/warpsize)/4;
        if (nitor == 0) nitor = 1;
        float res =0;
        #pragma unroll
        // mat = &mat[row*N];
        for (int i=0;i<nitor;i++)
        {
            float4 cur = FETCH_FLOAT4(mat[row*N+(i*warpsize+col)*4]);
            float4 cur1 = FETCH_FLOAT4(vector[(i*warpsize+col)*4]);
            res+=  cur.x*cur1.x;
            res+=  cur.y*cur1.y;
            res+=  cur.z*cur1.z;
            res+=  cur.w*cur1.w;
        }
        res = warp_reduce<warpsize>(res);
       if (col ==0) o_vector[row] = res;
    }
}

int main(int argc, char ** argv)
{
    if (argc!= 3)
    {
        printf("please input ./main M N");
        exit(0);
    }
    const int M = atoi(argv[1]);
    const int N = atoi(argv[2]);

    float * mat =(float*)malloc(M*N*sizeof(float));
    float * vector =(float*)malloc(N*sizeof(float));
    float * res =(float*)malloc(M*sizeof(float));
    float * res1 =(float*)malloc(M*sizeof(float));

    for (int i=0;i<M*N;i++)mat[i] = i/13;
    for (int i=0; i<N;i++) vector[i] = i/13;
    memset(res,0,M*sizeof(float));
    memset(res1,0,M*sizeof(float));

    float *d_mat;
    float *d_vector;
    float *d_res;
    

    cudaMalloc(&d_mat,M*N*sizeof(float));
    cudaMalloc(&d_vector,N*sizeof(float));
    cudaMalloc(&d_res,M*sizeof(float));

    cudaMemcpy(d_mat,mat,M*N*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector,vector,N*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_res,res,M*sizeof(float),cudaMemcpyHostToDevice);

    const int warpSize = 32;
    dim3 Grid(M/4);
    dim3 Block(32,4);
    int nIter = 1000;

    cudaEvent_t start ,stop;
    float ms = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int run = 0 ; run < nIter; run ++ ) {
        Sgemv<warpSize><<< Grid, Block >>>(d_mat, d_vector, d_res, M, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms,start,stop);
    printf("ms : %.2f \n",ms/1000);
    cudaMemcpy(res,d_res,M*sizeof(float),cudaMemcpyDeviceToHost);



    // cublas
    cublasHandle_t blas_handle;  
    cublasCreate(&blas_handle);
    float alpha = 1.0;
    float beta = 0;
    cudaMemcpy(d_res, res1, M*sizeof(float), cudaMemcpyHostToDevice);
    cudaEvent_t start1 ,stop1;
    float ms1 = 0;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    cudaEventRecord(start1);
    for (int run = 0 ; run < nIter; run ++ ) {
        cublasSgemv (blas_handle, CUBLAS_OP_T, 
            N, M, &alpha, 
            d_mat, N, d_vector, 1, &beta, d_res, 1
        );
    }
    cudaEventRecord(stop1);
    cudaEventSynchronize(stop1);
    cudaEventElapsedTime(&ms1,start1,stop1);
    printf("ms : %.2f \n",ms1/1000);
    cudaMemcpy( res1, d_res, M*sizeof(float), cudaMemcpyDeviceToHost);
    cublasDestroy(blas_handle); 
    
    double eps = 1.e-6;  // machine zero
    bool correct = true;
    for (int i = 0; i < M; i++) {
        double abs_err = fabs(res1[i] - res[i]);
        double dot_length = M;
        double abs_val = fabs(res[i]);
        double rel_err = abs_err / abs_val / dot_length;
        if (rel_err > eps) {
            printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n",
                    i, res[i], res1[i], eps);
            correct = false;
            break;
        }
    }

    printf("%s\n", correct ? "Result= PASS" : "Result= FAIL");

}