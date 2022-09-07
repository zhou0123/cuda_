#include <stdio.h>
#include <stdlib.h>
#include "assert.h" 

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FEATCH_FLOAT4(pointer)(reinterpret_cast<float4*>(&(pointer))[0])


template<
    const int BM,
    const int BK,
    const int BN,
    const int RM,
    const int RN,
    const bool double_buffer
>
__global__ void Sgemm(
    float* __restrict__ A,
    float* __restrict__ B,
    float* __restrict__ C,
    const int M,
    const int N,
    const int K
)
{
    int by =blockIdx.y;
    int bx =blockIdx.x;

    int ty =threadIdx.y;
    int tx =threadIdx.x;


    const int x_threads_size = BN/RN;
    const int y_threads_size = BM/RM;

    const int threads_per_block = x_threads_size*y_threads_size;
    
    const int tid = ty*x_threads_size + tx;

    const int A_nums_thread_per_line = BK/4;
    const int B_nums_thread_per_line = BN/4;

    const int A_stride = threads_per_block/A_nums_thread_per_line;
    const int B_stride = threads_per_block/B_nums_thread_per_line;

    const int A_start  = tid/A_nums_thread_per_line;
    const int A_col    = tid%A_nums_thread_per_line*4;
    const int B_start  = tid/B_nums_thread_per_line;
    const int B_col    = tid%B_nums_thread_per_line*4;

    const int A_times = BM*BK/(threads_per_block*4);
    const int B_times = BN*BK/(threads_per_block*4);

    float ldg_A[A_times*4];
    float ldg_B[B_times*4];

    float frag_A[2][RM];
    float frag_B[2][RN];
    float res[RM][RN] ={0};

    __shared__ float As[2][BK][BM];
    __shared__ float Bs[2][BK][BN];

    A = &A[BM*by*K];
    B = &B[BN*bx];
    #pragma unroll
    for (int i=0;i<BM;i+=A_stride)
    {
        int index = i/A_stride*4;
        FEATCH_FLOAT4(ldg_A[index]) = FEATCH_FLOAT4(A[OFFSET(
            A_start+i,
            A_col,
            K
        )]);
        As[0][A_col][A_start+i] = ldg_A[index];
        As[0][A_col+1][A_start+i] = ldg_A[index+1];
        As[0][A_col+2][A_start+i] = ldg_A[index+2];
        As[0][A_col+3][A_start+i] = ldg_A[index+3];
    }
    #pragma unroll
    for (int i=0;i<BK;i+=B_stride)
    {
        FEATCH_FLOAT4(Bs[0][B_start+i][B_col]) = FEATCH_FLOAT4(B[OFFSET(B_start+i,B_col,N)]);
    }
    __syncthreads();
    #pragma unroll
    for (int thread_y=0;thread_y<RM;thread_y+=4)
    {
        FEATCH_FLOAT4(frag_A[0][thread_y]) = FEATCH_FLOAT4(As[0][0][ty*RM+thread_y]);
    }
    #pragma unroll
    for (int thread_x=0;thread_x<RN;thread_x+=4)
    {
        FEATCH_FLOAT4(frag_B[0][thread_x]) = FEATCH_FLOAT4(Bs[0][0][tx*RN+thread_x]);
    }
    int write_stage_idx = 1;
    int tile_idx = 0;
    do {
        tile_idx+=BK;
        if (tile_idx<K)
        {
            #pragma unroll
            for (int i=0;i<BM;i+=A_stride)
            {
                int index = i/A_stride*4;
                FEATCH_FLOAT4(ldg_A[index]) = FEATCH_FLOAT4(A[OFFSET(
                    A_start+i,
                    A_col+tile_idx,
                    K
                )]);
            }
            #pragma unroll
            for (int i=0;i<BK;i+=B_stride)
            {
                int index = i/B_stride*4;
                FEATCH_FLOAT4(ldg_B[index]) = FEATCH_FLOAT4(B[OFFSET(
                    B_start+i+tile_idx,
                    B_col,
                    N
                )]);
            }
        }

        int load_stage_idx = write_stage_idx ^ 1;
        #pragma unroll
        for (int j=0;j<BK-1;j++)
        {
            #pragma unroll
            for (int thread_y=0;thread_y<RM;thread_y+=4)
            {
                FEATCH_FLOAT4(frag_A[(j+1)%2][thread_y]) = FEATCH_FLOAT4(As[load_stage_idx][j+1][thread_y+ty*RM]); 
            }
            #pragma unroll
            for (int thread_x=0;thread_x<RN;thread_x+=4)
            {
                FEATCH_FLOAT4(frag_B[(j+1)%2][thread_x]) = FEATCH_FLOAT4(Bs[load_stage_idx][j+1][thread_x+tx*RN]);
            }
            #pragma unroll
            for (int thread_y=0;thread_y<RM;thread_y++)
            {
                #pragma unroll
                for (int thread_x= 0;thread_x<RN;thread_x++)
                {
                    res[thread_y][thread_x] += frag_A[j%2][thread_y]*frag_B[j%2][thread_x];
                }
            }
        }
        if (tile_idx<K)
        {
            #pragma unroll
            for (int i=0;i<BM;i+=A_stride)
            {
                int index = i/A_stride*4;
                As[write_stage_idx][A_col][A_start+i] = ldg_A[index];
                As[write_stage_idx][A_col+1][A_start+i] = ldg_A[index+1];
                As[write_stage_idx][A_col+2][A_start+i] = ldg_A[index+2];
                As[write_stage_idx][A_col+3][A_start+i] = ldg_A[index+3];
            }
            #pragma unroll
            for (int i=0;i<BK;i+=B_stride)
            {
                int index = i/B_stride*4;
                FEATCH_FLOAT4(Bs[write_stage_idx][B_start+i][B_col]) = FEATCH_FLOAT4(ldg_B[index]);
            }
             __syncthreads();
            write_stage_idx ^= 1;
        }

        for (int thread_y=0;thread_y<RM;thread_y+=4)
        {
            FEATCH_FLOAT4(frag_A[0][thread_y]) = FEATCH_FLOAT4(As[load_stage_idx^1][0][thread_y+ty*RM]); 
        }
        #pragma unroll
        for (int thread_x=0;thread_x<RN;thread_x+=4)
        {
            FEATCH_FLOAT4(frag_B[0][thread_x]) = FEATCH_FLOAT4(Bs[load_stage_idx^1][0][thread_x+tx*RN]);
        }
        #pragma unroll
        for (int thread_y=0;thread_y<RM;thread_y++)
        {
            #pragma unroll
            for (int thread_x= 0;thread_x<RN;thread_x++)
            {
                res[thread_y][thread_x] += frag_A[1][thread_y]*frag_B[1][thread_x];
            }
        }
    }while(tile_idx<K);

    #pragma unroll
    for (int thread_y=0;thread_y<RM;thread_y++)
    {
        #pragma unroll
        for (int thread_x= 0;thread_x<RN;thread_x+=4)
        {
            FEATCH_FLOAT4(C[OFFSET(
                by*BM+ty*RM+thread_y,
                bx*BN+tx*RN+thread_x,
                N
            )])=FEATCH_FLOAT4(res[thread_y][thread_x]);
        }
    }   

}



int main(int argc,char ** argv)
{
    if (argc != 4)
    {
        printf("please input ./main M K N");
        exit(0);
    }

    const int M = atoi(argv[1]);
    const int K = atoi(argv[2]);
    const int N = atoi(argv[3]);
    assert( M%8 == 0); 
    assert( N%8 == 0); 
    assert( K%8 == 0); 

    float bytes_A = sizeof(float)*M*K;
    float bytes_B = sizeof(float)*K*N;
    float bytes_C = sizeof(float)*M*N;

    float *A = (float*) malloc(bytes_A*sizeof(float));
    float *B = (float*) malloc(bytes_B*sizeof(float));
    float *C = (float*) malloc(bytes_C*sizeof(float));
    float *C1 = (float*) malloc(bytes_C*sizeof(float));


    for(int i=0;i<bytes_A;i++) A[i] = i/13;
    for(int i=0;i<bytes_B;i++) B[i] = i%13;

    float*d_A;
    float*d_B;
    float*d_C;
    cudaMalloc(&d_A,bytes_A);
    cudaMalloc(&d_B,bytes_B);
    cudaMalloc(&d_C,bytes_C);

    cudaMemcpy(d_A,A,bytes_A,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,B,bytes_B,cudaMemcpyHostToDevice);
    cudaMemcpy(d_C,C,bytes_C,cudaMemcpyHostToDevice);

    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int RM = 8;
    const int RN = 8;
    const bool DOUBLE_BUFFER =true;
    double msecPerMatrixMul[2] = {0, 0};
    double gigaFlops[2] = {0, 0};
    double flopsPerMatrixMul = 2.0 * M * N * K;

    float ms_total =0;
    int niter =1000;

    cudaEvent_t start , stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int i=0;i<niter;i++)
    {
        dim3 Grid(N/BN,M/BM);
        dim3 Block(BN/RN,BM/RM);
        Sgemm<BM,BK,BN,RM,RN,DOUBLE_BUFFER><<<Grid,Block>>>(d_A,d_B,d_C,M,N,K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_total, start, stop);
    cudaMemcpy(C,d_C,bytes_C,cudaMemcpyDeviceToHost);

    msecPerMatrixMul[0] = ms_total / niter;
    gigaFlops[0] = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul[0] / 1000.0f);
    printf( "My gemm Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n",
        gigaFlops[0],
        msecPerMatrixMul[0],
        flopsPerMatrixMul);

    // cublas
    cublasHandle_t blas_handle;  
    cublasCreate(&blas_handle);
    float alpha = 1.0;
    float beta = 0;
    cudaMemcpy(d_C, C, bytes_C, cudaMemcpyHostToDevice);
    cudaEventRecord(start);
    for (int run = 0 ; run < niter; run ++ ) {
        cublasSgemm (blas_handle, CUBLAS_OP_T, CUBLAS_OP_T, 
            M, N, K, &alpha, 
            d_A, K, d_B, N, &beta, d_C, N
        );
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_total, start, stop);

    cudaMemcpy(C1, d_C, bytes_C, cudaMemcpyDeviceToHost);

    msecPerMatrixMul[1] = ms_total / niter;
    gigaFlops[1] = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul[1] / 1000.0f);
    printf( "CuBlas Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n",
        gigaFlops[1],
        msecPerMatrixMul[1],
        flopsPerMatrixMul);

    cublasDestroy(blas_handle); 
    
    double eps = 1.e-6;  
    bool correct = true;
    for (int i = 0; i < M * N; i++) {
        int row = i / N;
        int col = i % N;
        double abs_err = fabs(C[i] - C1[col * M + row]);
        double dot_length = M;
        double abs_val = fabs(C[i]);
        double rel_err = abs_err / abs_val / dot_length;
        if (rel_err > eps) {
            printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n",
                    i, C[i], C1[col * M + row], eps);
            correct = false;
            break;
        }
    }

    printf("%s\n", correct ? "Result= PASS" : "Result= FAIL");
    printf("ratio= %f\n", gigaFlops[0] / gigaFlops[1]);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    free(A);
    free(B);
    free(C);
    free(C1);


}