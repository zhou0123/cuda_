#include<stdio.h>
#include<stdlib.h>
#include"assert.h"
#include<iostream>
#include<cuda_runtime.h>
#include<cublas_v2.h>

#define OFFSET(row,col,ld) (row*ld+col)
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

#define checkCudaErrors(func) \
{\
    cudaError_t e = (func);\
    if (e!= cudaSuccess)\
    printf("%s %d CUDA : %s \n",__FILE__,__LINE__,cudaGetErrorString(e));\
}

/*template<
    const int BLOCK_SIZE_M,
    const int BLOCK_SIZE_K,
    const int BLOCK_SIZE_N,
    const int THREAD_SIZE_Y,
    const int THREAD_SIZE_X,
    const bool ENABLE_DOUBLE_BUFFER
>
__global__ void Sgemm(
    float * __restrict__ A,
    float * __restrict__ B,
    float * __restrict__ C,
    const int M,
    const int N,
    const int K

)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    const int THREAD_X_PER_BLOCK = BLOCK_SIZE_N/THREAD_SIZE_X;
    const int THREAD_Y_PER_BLOCK = BLOCK_SIZE_M/THREAD_SIZE_Y;
    const int THREAD_NUM_PER_BLOCK = THREAD_X_PER_BLOCK * THREAD_Y_PER_BLOCK;

    const int tid = threadIdx.y*THREAD_X_PER_BLOCK+threadIdx.x;

    //shared_memory
    __shared__ float As[2][BLOCK_SIZE_K][BLOCK_SIZE_M];
    __shared__ float Bs[2][BLOCK_SIZE_K][BLOCK_SIZE_N];

    float accum[THREAD_SIZE_Y][THREAD_SIZE_X] = {0};

    float frag_a[2][THREAD_SIZE_Y];
    float frag_b[2][THREAD_SIZE_X];

    const int ldg_num_a = BLOCK_SIZE_M*BLOCK_SIZE_K/(THREAD_NUM_PER_BLOCK*4);
    const int ldg_num_b = BLOCK_SIZE_K*BLOCK_SIZE_N/(THREAD_NUM_PER_BLOCK*4);

    float ldg_a_reg[4*ldg_num_a];
    float ldg_b_reg[4*ldg_num_b];

    const int A_TILE_THREAD_PER_ROW =BLOCK_SIZE_K/4;
    const int B_TILE_THREAD_PER_ROW =BLOCK_SIZE_N/4;

    const int A_TILE_ROW_START = tid/A_TILE_THREAD_PER_ROW;
    const int B_TILE_ROW_SATRT = tid/B_TILE_THREAD_PER_ROW;

    const int A_TILE_COL = tid%A_TILE_THREAD_PER_ROW*4;
    const int B_TILE_COL = tid%B_TILE_THREAD_PER_ROW*4;

    const int A_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK/A_TILE_THREAD_PER_ROW;
    const int B_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK/B_TILE_THREAD_PER_ROW;

    A = &A[BLOCK_SIZE_M*by*K];
    B = &B[BLOCK_SIZE_N*bx];
    #pragma unroll
    for (int i=0;i<BLOCK_SIZE_M;i+=A_TILE_ROW_STRIDE)
    {
        int index = i/A_TILE_ROW_STRIDE*4;
        FETCH_FLOAT4(ldg_a_reg[index]) = FETCH_FLOAT4(A[OFFSET(
                i+A_TILE_ROW_START,
                A_TILE_COL,
                K)]);
        As[0][A_TILE_COL][i+A_TILE_ROW_START] = ldg_a_reg[index];
        As[0][A_TILE_COL+1][i+A_TILE_ROW_START] = ldg_a_reg[index+1];
        As[0][A_TILE_COL+2][i+A_TILE_ROW_START] = ldg_a_reg[index+2];
        As[0][A_TILE_COL+3][i+A_TILE_ROW_START] = ldg_a_reg[index+3];
    }
    #pragma unroll
    for (int i=0;i<BLOCK_SIZE_N;i+=B_TILE_ROW_STRIDE)
    {
        FETCH_FLOAT4(Bs[0][i+B_TILE_ROW_SATRT][B_TILE_COL])=FETCH_FLOAT4(
            B[OFFSET(
                i+B_TILE_ROW_SATRT,
                B_TILE_COL,
                N
            )]
        );
    }
    __syncthreads();
    #pragma unrool
    for (int thread_y=0;thread_y<THREAD_SIZE_Y;thread_y+=4)
    {
        FETCH_FLOAT4(frag_a[0][thread_y]) = FETCH_FLOAT4(As[0][0][THREAD_SIZE_Y * ty+thread_y]);
    }
    for (int thread_x=0;thread_x<THREAD_SIZE_X;thread_x+=4)
    {
        FETCH_FLOAT4(frag_b[0][thread_x]) = FETCH_FLOAT4(Bs[0][0][THREAD_SIZE_X * tx+thread_x]);
    }
    int write_stage_idx = 1;
    int tile_idx = 0;
    do{
        tile_idx += BLOCK_SIZE_K;
        if (tile_idx<K)
        {
            #pragma unroll
            for (int i=0;i<BLOCK_SIZE_M;i+=A_TILE_ROW_STRIDE)
            {
                int index = i/A_TILE_ROW_STRIDE*4;
                FETCH_FLOAT4(ldg_a_reg[index]) = FETCH_FLOAT4(A[OFFSET(
                    i+A_TILE_ROW_START,
                    A_TILE_COL+tile_idx,
                    K
                )]);
            }
            
            #pragma unroll
            for (int i=0;i<BLOCK_SIZE_N;i+=B_TILE_ROW_STRIDE)
            {
                int index = i/B_TILE_ROW_STRIDE*4;
                FETCH_FLOAT4(ldg_b_reg[index]) = FETCH_FLOAT4(B[OFFSET(
                    i+B_TILE_ROW_SATRT+tile_idx,
                    B_TILE_COL,
                    N
                )]);

            }
        }
        int load_stage_idx = write_stage_idx^1;
        #pragma unroll
        
        for (int j=0;j<BLOCK_SIZE_K-1;j++)
        {
            #pragma unroll
            for (int thread_y =0;thread_y<THREAD_SIZE_Y;thread_y+=4)
            {
                FETCH_FLOAT4(frag_a[(j+1)%2]) = FETCH_FLOAT4(As[load_stage_idx][j+1][ty*THREAD_SIZE_Y+thread_y]);
            }
            #pragma unroll
            for (int thread_x=0;thread_x<THREAD_SIZE_X;thread_x+=4)
            {
                FETCH_FLOAT4(frag_b[(j+1)%2]) = FETCH_FLOAT4(Bs[load_stage_idx][j+1][tx*THREAD_SIZE_X+thread_x]);
            }
            #pragma unroll
            for (int thread_y =0;thread_y<THREAD_SIZE_Y;thread_y++)
            {
                #pragma unroll
                for (int thread_x = 0;thread_x<THREAD_SIZE_X;thread_x++)
                {
                    accum[thread_y][thread_x] += frag_a[j%2][thread_y]*frag_a[j%2][thread_x];
                }
            }
        }
        if (tile_idx<K)
        {
            #pragma unroll
            for (int i=0;i<BLOCK_SIZE_M;i+=A_TILE_ROW_STRIDE)
            {
                int index = i/A_TILE_ROW_STRIDE*4;
                As[write_stage_idx][A_TILE_COL][A_TILE_ROW_START+i] = ldg_a_reg[index];
                As[write_stage_idx][A_TILE_COL+1][A_TILE_ROW_START+i] = ldg_a_reg[index+1];
                As[write_stage_idx][A_TILE_COL+2][A_TILE_ROW_START+i] = ldg_a_reg[index+2];
                As[write_stage_idx][A_TILE_COL+3][A_TILE_ROW_START+i] = ldg_a_reg[index+3];
            }
            #pragma unroll
            for (int i=0;i<BLOCK_SIZE_K;i+=B_TILE_ROW_STRIDE)
            {
                int index = i/B_TILE_ROW_STRIDE*4;
                FETCH_FLOAT4(Bs[write_stage_idx][B_TILE_ROW_SATRT+i][B_TILE_COL])=FETCH_FLOAT4(ldg_b_reg[index]);
            }
            __syncthreads();
            write_stage_idx ^=1;
        }
        #pragma unroll
        for (int thread_y =0;thread_y<THREAD_SIZE_Y;thread_y+=4)
        {
            FETCH_FLOAT4(frag_a[0][thread_y]) = FETCH_FLOAT4(As[load_stage_idx^1][0][THREAD_SIZE_Y*ty+thread_y]);
        }
        #pragma unroll
        for (int thread_x=0;thread_x<THREAD_SIZE_X;thread_x+=4)
        {
            FETCH_FLOAT4(frag_b[0][thread_x]) = FETCH_FLOAT4(Bs[load_stage_idx^1][0][THREAD_SIZE_X*tx+thread_x]);
        }
        #pragma unroll
        for (int thread_y=0;thread_y<THREAD_SIZE_Y;thread_y++)
        {
            #pragma unroll
            for (int thread_x=0;thread_x<THREAD_SIZE_X;thread_x++)
            {
                accum[thread_y][thread_x] += frag_a[1][thread_y]*frag_a[1][thread_x];
            }
        }
        }while(tile_idx<K);
        #pragma unroll
        for (int thread_y =0;thread_y<THREAD_SIZE_Y;thread_y++)
        {
            #pragma unroll
            for (int thread_x=0;thread_x<THREAD_SIZE_X;thread_x+=4)
            {
                FETCH_FLOAT4(C[OFFSET(
                    BLOCK_SIZE_M*by+THREAD_SIZE_Y*ty+thread_y,
                    BLOCK_SIZE_N*bx+THREAD_SIZE_X*tx+thread_x,
                    N
                )]) = FETCH_FLOAT4(accum[thread_y][thread_x]);
            }
        }
}*/
template <
    const int BLOCK_SIZE_M,  // height of block of C that each thread block calculate
    const int BLOCK_SIZE_K,  // width of block of A that each thread block load into shared memory
    const int BLOCK_SIZE_N,  // width of block of C that each thread block calculate
    const int THREAD_SIZE_Y, // height of block of C that each thread calculate
    const int THREAD_SIZE_X,  // width of block of C that each thread calculate
    const bool ENABLE_DOUBLE_BUFFER // whether enable double buffering or not
    > 
__global__ void Sgemm( 
    float * __restrict__ A,
    float * __restrict__ B,
    float * __restrict__ C, 
    const int M,
    const int N,
    const int K) {
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // the threads number in Block of X,Y
    const int THREAD_X_PER_BLOCK = BLOCK_SIZE_N / THREAD_SIZE_X;
    const int THREAD_Y_PER_BLOCK = BLOCK_SIZE_M / THREAD_SIZE_Y;
    const int THREAD_NUM_PER_BLOCK = THREAD_X_PER_BLOCK * THREAD_Y_PER_BLOCK;

    // thread id in cur Block
    const int tid = ty * THREAD_X_PER_BLOCK + tx;

    // shared memory
    __shared__ float As[2][BLOCK_SIZE_K][BLOCK_SIZE_M];
    __shared__ float Bs[2][BLOCK_SIZE_K][BLOCK_SIZE_N];
    // registers for C
    float accum[THREAD_SIZE_Y][THREAD_SIZE_X] = {0};
    // registers for A and B
    float frag_a[2][THREAD_SIZE_Y];
    float frag_b[2][THREAD_SIZE_X];
    // registers load global memory
    const int ldg_num_a = BLOCK_SIZE_M * BLOCK_SIZE_K / (THREAD_NUM_PER_BLOCK * 4);
    const int ldg_num_b = BLOCK_SIZE_K * BLOCK_SIZE_N / (THREAD_NUM_PER_BLOCK * 4);
    float ldg_a_reg[4*ldg_num_a];
    float ldg_b_reg[4*ldg_num_b];

    // threads number in one row
    const int A_TILE_THREAD_PER_ROW = BLOCK_SIZE_K / 4;
    const int B_TILE_THREAD_PER_ROW = BLOCK_SIZE_N / 4;

    // row number and col number that needs to be loaded by this thread
    const int A_TILE_ROW_START = tid / A_TILE_THREAD_PER_ROW;
    const int B_TILE_ROW_START = tid / B_TILE_THREAD_PER_ROW;

    const int A_TILE_COL = tid % A_TILE_THREAD_PER_ROW * 4; 
    const int B_TILE_COL = tid % B_TILE_THREAD_PER_ROW * 4;

    // row stride that thread uses to load multiple rows of a tile
    const int A_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / A_TILE_THREAD_PER_ROW;
    const int B_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / B_TILE_THREAD_PER_ROW;

    A = &A[(BLOCK_SIZE_M * by)* K];
    B = &B[BLOCK_SIZE_N * bx];

    //transfer first tile from global mem to shared mem
    // load A from global memory to shared memory
    #pragma unroll
    for ( int i = 0 ; i < BLOCK_SIZE_M ; i += A_TILE_ROW_STRIDE) {
        int ldg_index = i / A_TILE_ROW_STRIDE * 4;
        FETCH_FLOAT4(ldg_a_reg[ldg_index]) = FETCH_FLOAT4(A[OFFSET(
            A_TILE_ROW_START + i, // row
            A_TILE_COL, // col
            K )]);
        As[0][A_TILE_COL][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index];
        As[0][A_TILE_COL+1][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index+1];
        As[0][A_TILE_COL+2][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index+2];
        As[0][A_TILE_COL+3][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index+3];
    }
    // load B from global memory to shared memory
    #pragma unroll
    for ( int i = 0 ; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
        FETCH_FLOAT4(Bs[0][B_TILE_ROW_START + i][B_TILE_COL]) = FETCH_FLOAT4(B[OFFSET(
                B_TILE_ROW_START + i, // row
                B_TILE_COL, // col
                N )]);
    }
    __syncthreads();
    // load A from shared memory to register
    #pragma unroll
    for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y += 4) {
        FETCH_FLOAT4(frag_a[0][thread_y]) = FETCH_FLOAT4(As[0][0][THREAD_SIZE_Y * ty + thread_y]);
    }
    // load B from shared memory to register
    #pragma unroll
    for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4) {
        FETCH_FLOAT4(frag_b[0][thread_x]) = FETCH_FLOAT4(Bs[0][0][THREAD_SIZE_X * tx + thread_x]);
    }

    int write_stage_idx = 1;
    int tile_idx = 0;
    do{
        tile_idx += BLOCK_SIZE_K;
        // load next tile from global mem
        if(tile_idx< K){
            #pragma unroll
            for ( int i = 0 ; i < BLOCK_SIZE_M ; i += A_TILE_ROW_STRIDE) {
                int ldg_index = i / A_TILE_ROW_STRIDE * 4;
                FETCH_FLOAT4(ldg_a_reg[ldg_index]) = FETCH_FLOAT4(A[OFFSET(
                    A_TILE_ROW_START + i, // row
                    A_TILE_COL + tile_idx, // col
                    K )]);
            }
            #pragma unroll
            for ( int i = 0 ; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
                int ldg_index = i / B_TILE_ROW_STRIDE * 4;
                FETCH_FLOAT4(ldg_b_reg[ldg_index]) = FETCH_FLOAT4(B[OFFSET(
                    tile_idx + B_TILE_ROW_START + i, // row
                    B_TILE_COL, // col
                    N )]);
            }
        }

        int load_stage_idx = write_stage_idx ^ 1;

        #pragma unroll
        for(int j=0; j<BLOCK_SIZE_K-1; ++j){
            // load next tile from shared mem to register 
            // load A from shared memory to register
            #pragma unroll
            for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y += 4) {
                FETCH_FLOAT4(frag_a[(j+1)%2][thread_y]) = FETCH_FLOAT4(As[load_stage_idx][j+1][THREAD_SIZE_Y * ty + thread_y]);
            }
            // load B from shared memory to register
            #pragma unroll
            for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4) {
                FETCH_FLOAT4(frag_b[(j+1)%2][thread_x]) = FETCH_FLOAT4(Bs[load_stage_idx][j+1][THREAD_SIZE_X * tx + thread_x]);
            }
            // compute C THREAD_SIZE_X x THREAD_SIZE_Y
            #pragma unroll
            for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
                #pragma unroll
                for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
                    accum[thread_y][thread_x] += frag_a[j%2][thread_y] * frag_b[j%2][thread_x];
                }
            }
        }

        if(tile_idx < K){
            #pragma unroll
            for ( int i = 0 ; i < BLOCK_SIZE_M ; i += A_TILE_ROW_STRIDE) {
                int ldg_index = i / A_TILE_ROW_STRIDE * 4;
                As[write_stage_idx][A_TILE_COL][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index];
                As[write_stage_idx][A_TILE_COL+1][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index+1];
                As[write_stage_idx][A_TILE_COL+2][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index+2];
                As[write_stage_idx][A_TILE_COL+3][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index+3];
            }
            // load B from global memory to shared memory
            #pragma unroll
            for ( int i = 0 ; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
                int ldg_index = i / B_TILE_ROW_STRIDE * 4;
                FETCH_FLOAT4(Bs[write_stage_idx][B_TILE_ROW_START + i][B_TILE_COL]) = FETCH_FLOAT4(ldg_b_reg[ldg_index]);
            }
            // use double buffer, only need one sync
            __syncthreads();
            // switch
            write_stage_idx ^= 1;
        }

        // load first tile from shared mem to register of next iter
        // load A from shared memory to register
        #pragma unroll
        for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y += 4) {
            FETCH_FLOAT4(frag_a[0][thread_y]) = FETCH_FLOAT4(As[load_stage_idx^1][0][THREAD_SIZE_Y * ty + thread_y]);
        }
        // load B from shared memory to register
        #pragma unroll
        for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4) {
            FETCH_FLOAT4(frag_b[0][thread_x]) = FETCH_FLOAT4(Bs[load_stage_idx^1][0][THREAD_SIZE_X * tx + thread_x]);
        }
        //compute last tile mma THREAD_SIZE_X x THREAD_SIZE_Y
        #pragma unroll
        for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
            #pragma unroll
            for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
                accum[thread_y][thread_x] += frag_a[1][thread_y] * frag_b[1][thread_x];
            }
        }
    }while(tile_idx< K);

    // store back to C
    #pragma unroll
    for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
        #pragma unroll
        for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x+=4) {
            FETCH_FLOAT4(C[OFFSET(
                BLOCK_SIZE_M * by + ty * THREAD_SIZE_Y + thread_y,
                BLOCK_SIZE_N * bx + tx * THREAD_SIZE_X + thread_x,
                N)]) = FETCH_FLOAT4(accum[thread_y][thread_x]);
        }
    }
}

int main(int argc ,char ** argv)
{
    // if (argc!=4) 
    // {
    //     printf("usage: ./main [M],[K],[N]");
    //     exit(0);
    // }

    // size_t M = atoi(argv[1]);
    // size_t K = atoi(argv[2]);
    // size_t N = atoi(argv[3]);

    // assert(M%8==0);
    // assert(K%8==0);
    // assert(N%8==0);


    // size_t bytes_A  = sizeof(float)*M*K;
    // size_t bytes_B  = sizeof(float)*N*K;
    // size_t bytes_C  = sizeof(float)*M*N;
    // float* A = (float*) malloc(bytes_A);
    // float* B = (float*) malloc (bytes_B);
    // float* C = (float*) malloc(bytes_C);
    // float* C1 =(float*) malloc(bytes_C);

    // float * d_A;
    // float * d_B;
    // float * d_C;
    // cudaMalloc(&d_A,bytes_A);
    // cudaMalloc(&d_B,bytes_B);
    // cudaMalloc(&d_C,bytes_C);

    // double msecPerMatrixMul[2] = {0,0};
    // double gigaFlops[2] = {0, 0};
    // double flopsPerMatrixMul = 2.0 * M * N * K;

    // const int BM = 128;
    // const int BK=8;
    // const int BN = 128;
    // const int THREAD_SIZE_X =8;
    // const int THREAD_SIZE_Y =8;
    // const bool DOUBLE_BUFFER = false;

    // for (int i=0;i<M*K;i++)
    // {
    //     A[i] = i/13;
    // }
    // for (int i=0;i<K*N;i++)
    // {
    //     B[i] = i%13;
    // }
    // cudaMemcpy(d_A,A,bytes_A,cudaMemcpyHostToDevice);
    // cudaMemcpy(d_B,B,bytes_B,cudaMemcpyHostToDevice);
    // cudaEvent_t start,stop;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);
    // float mseTotal =0;
    // int niter = 1000;
    // cudaMemcpy(d_C,C,bytes_C,cudaMemcpyHostToDevice);
    // cudaEventRecord(start);
    // for (int run=0;run<niter;run++)
    // {
    //     dim3 Block(BN/THREAD_SIZE_X,BM/THREAD_SIZE_Y);
    //     dim3 Grid(N/BN,M/BM);

    //     Sgemm<BM,BK,BN,THREAD_SIZE_Y,THREAD_SIZE_X,DOUBLE_BUFFER><<<Grid,Block>>>(d_A,d_B,d_C,M,N,K);
    // }
    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&mseTotal,start,stop);
    // cudaMemcpy(C,d_C,bytes_C,cudaMemcpyDeviceToHost);
    // msecPerMatrixMul[0] = mseTotal / niter;
    // gigaFlops[0] = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul[0] / 1000.0f);
    // printf( "My gemm Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n",
    //     gigaFlops[0],
    //     msecPerMatrixMul[0],
    //     flopsPerMatrixMul);
    if (argc != 4) {
        printf("usage: ./main [M] [K] [N]\n");
        exit(0);
    }
    size_t M = atoi(argv[1]);
    size_t K = atoi(argv[2]);
    size_t N = atoi(argv[3]);

    assert( M%8 == 0); 
    assert( N%8 == 0); 
    assert( K%8 == 0); 

    size_t bytes_A = sizeof(float) * M * K;
    size_t bytes_B = sizeof(float) * K * N;
    size_t bytes_C = sizeof(float) * M * N;
    float* h_A = (float*)malloc(bytes_A);
    float* h_B = (float*)malloc(bytes_B);
    float* h_C = (float*)malloc(bytes_C);
    float* h_C1 = (float*)malloc(bytes_C);

    float* d_A;
    float* d_B;
    float* d_C;

    checkCudaErrors(cudaMalloc(&d_A, bytes_A));
    checkCudaErrors(cudaMalloc(&d_B, bytes_B));
    checkCudaErrors(cudaMalloc(&d_C, bytes_C));
    double msecPerMatrixMul[2] = {0, 0};
    double gigaFlops[2] = {0, 0};
    double flopsPerMatrixMul = 2.0 * M * N * K;

    const int BLOCK_SIZE_M = 128;
    const int BLOCK_SIZE_K = 8;
    const int BLOCK_SIZE_N = 128;
    const int THREAD_SIZE_X = 8;
    const int THREAD_SIZE_Y = 8;
    const bool ENABLE_DOUBLE_BUFFER = false;

    // generate A
    for( int i = 0; i < M * K; i++ ){
        h_A[i] = i / 13;
    }

    // generate B
    for( int i = 0; i < K * N; i++ ) {
        h_B[i] = i % 13;
    }

    checkCudaErrors(cudaMemcpy( d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy( d_B, h_B, bytes_B, cudaMemcpyHostToDevice));
    
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    float msecTotal = 0;
    int nIter = 1000;

    checkCudaErrors(cudaMemcpy( d_C, h_C, bytes_C, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaEventRecord(start));
    for (int run = 0 ; run < nIter; run ++ ) {
        dim3 dimBlock(BLOCK_SIZE_N / THREAD_SIZE_X, BLOCK_SIZE_M / THREAD_SIZE_Y);
        dim3 dimGrid(N / BLOCK_SIZE_N, M / BLOCK_SIZE_M);
        Sgemm<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_Y, THREAD_SIZE_X, ENABLE_DOUBLE_BUFFER> 
        <<< dimGrid, dimBlock >>>(d_A, d_B, d_C, M, N, K);
    }
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
    checkCudaErrors(cudaMemcpy( h_C, d_C, bytes_C, cudaMemcpyDeviceToHost));

    msecPerMatrixMul[0] = msecTotal / nIter;
    gigaFlops[0] = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul[0] / 1000.0f);
    printf( "My gemm Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n",
        gigaFlops[0],
        msecPerMatrixMul[0],
        flopsPerMatrixMul);
    

    // // cublas
    // cublasHandle_t blas_handle;  
    // cublasCreate(&blas_handle);
    // float alpha = 1.0;
    // float beta = 0;
    // cudaMemcpy( d_C, C1, bytes_C, cudaMemcpyHostToDevice);
    // cudaEventRecord(start);
    // for (int run = 0 ; run < niter; run ++ ) {
    //     cublasSgemm (blas_handle, CUBLAS_OP_T, CUBLAS_OP_T, 
    //         M, N, K, &alpha, 
    //         d_A, K, d_B, N, &beta, d_C, N
    //     );
    // }
    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&mseTotal, start, stop);

    // cudaMemcpy(C1, d_C, bytes_C, cudaMemcpyDeviceToHost);

    // msecPerMatrixMul[1] = mseTotal / niter;
    // gigaFlops[1] = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul[1] / 1000.0f);
    // printf( "CuBlas Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n",
    //     gigaFlops[1],
    //     msecPerMatrixMul[1],
    //     flopsPerMatrixMul);

    // cublasDestroy(blas_handle); 
    
    // double eps = 1.e-6;  // machine zero
    // bool correct = true;
    // for (int i = 0; i < M * N; i++) {
    //     int row = i / N;
    //     int col = i % N;
    //     double abs_err = fabs(C[i] - C1[col * M + row]);
    //     double dot_length = M;
    //     double abs_val = fabs(C[i]);
    //     double rel_err = abs_err / abs_val / dot_length;
    //     if (rel_err > eps) {
    //         printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n",
    //                 i, C[i], C1[col * M + row], eps);
    //         correct = false;
    //         break;
    //     }
    // }

    // printf("%s\n", correct ? "Result= PASS" : "Result= FAIL");
    // printf("ratio= %f\n", gigaFlops[0] / gigaFlops[1]);
    
    // Free Memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    // free(A);
    // free(B);
    // free(C);
    // free(C1);
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C1);

}
