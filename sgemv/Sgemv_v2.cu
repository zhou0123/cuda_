#include <stdio.h>
#include <stdlib.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h> 

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

// if N <= 16
template <
    const int ROW_PER_WARP
    > 
__global__ void Sgemv_v2( 
    float * __restrict__ A,
    float * __restrict__ x,
    float * __restrict__ y, 
    const int M,
    const int N) {
    int bx = blockIdx.x;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    const int warp_size=32;
    int laneId= tx % warp_size;
    int current_warp_row = (blockDim.y * bx + ty) * ROW_PER_WARP;
    const int kWarp_size = warp_size / ROW_PER_WARP;
    int kLaneId = laneId % kWarp_size;
    int current_thread_row = current_warp_row + laneId / kWarp_size;

    if(current_thread_row < M){
        float res=0;
        int current_col = kLaneId;
        res += A[current_thread_row * N + current_col] * x[current_col];
        res = warp_reduce<kWarp_size>(res);
        if(kLaneId==0) y[current_thread_row]=res;
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
    const int ROW_PER_WARP = 2;
    

    cudaMalloc(&d_mat,M*N*sizeof(float));
    cudaMalloc(&d_vector,N*sizeof(float));
    cudaMalloc(&d_res,M*sizeof(float));

    cudaMemcpy(d_mat,mat,M*N*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector,vector,N*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_res,res,M*sizeof(float),cudaMemcpyHostToDevice);

    dim3 Grid(M/4);
    dim3 Block(32,4);
    int nIter = 1000;

    cudaEvent_t start ,stop;
    float ms = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int run = 0 ; run < nIter; run ++ ) {
        Sgemv_v2<ROW_PER_WARP><<< Grid, Block >>>(d_mat, d_vector, d_res, M, N);
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