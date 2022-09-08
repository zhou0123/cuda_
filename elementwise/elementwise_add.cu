#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <sys/time.h>

#define THREAD_PER_BLOCK 256
#define FETCH_FLOAT2(pointer)(reinterpret_cast<float2*>(&(pointer))[0])
#define FETCH_FLOAT4(pointer)(reinterpret_cast<float4*>(&(pointer))[0])
__global__ void elementwise_add_2 (float* d_A,float *d_B,float *d_C)
{
    const int tid = (blockIdx.x*blockDim.x+threadIdx.x)*2;
    float2 reg_a = FETCH_FLOAT2(d_A[tid]);
    float2 reg_b = FETCH_FLOAT2(d_B[tid]);
    float2 reg_c;
    reg_c.x = reg_a.x + reg_b.x;
    reg_c.y = reg_a.y + reg_b.y;
    FETCH_FLOAT2(d_C[tid]) = reg_c;
}
__global__ void elementwise_add(float* d_A,float *d_B,float *d_C)
{
    const int tid = (blockIdx.x * blockDim.x + threadIdx.x)*4;
    float4 reg_a = FETCH_FLOAT4(d_A[tid]);
    float4 reg_b = FETCH_FLOAT4(d_B[tid]);
    float4 reg_c;
    reg_c.x = reg_a.x+reg_b.x;
    reg_c.y = reg_a.y+reg_b.y;
    reg_c.z = reg_a.z+reg_b.z;
    reg_c.w = reg_a.w+reg_b.w;
    FETCH_FLOAT4(d_C[tid]) = reg_c;   
}
bool check(float* a ,float * b ,int nums)
{
    for (int i=0;i<nums;i++)
    {
        if (a[i]!=b[i])return false;
    }
    return true;
}
int main()
{
    const int N=32*1024*1024;
    float * A =  (float*)malloc(N*sizeof(float));
    float * B =  (float*)malloc(N*sizeof(float));
    float * C =  (float*)malloc(N*sizeof(float));
    float * C1 =  (float*)malloc(N*sizeof(float));

    for (int i=0;i<N;i++)
    {
        A[i] = i/13;
        B[i] = i%13;
        C[i] = i/13 + i%13;
    }

    float * d_A ;
    float * d_B ;
    float * d_C ;

    cudaMalloc(&d_A,N*sizeof(float));
    cudaMalloc(&d_B,N*sizeof(float));
    cudaMalloc(&d_C,N*sizeof(float));
    cudaMemcpy(d_A,A,N*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,B,N*sizeof(float),cudaMemcpyHostToDevice);
    cudaEvent_t start ,stop;
    float ms = 0;
    dim3 Grid (N/THREAD_PER_BLOCK/4,1);
    dim3 Block(THREAD_PER_BLOCK,1);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i=0;i<1000;i++)
    {
        elementwise_add<<<Grid,Block>>>(d_A,d_B,d_C);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms,start,stop);
    printf("ms : %.2f \n",ms/1000);



    cudaMemcpy(C1,d_C,N*sizeof(float),cudaMemcpyDeviceToHost);
    if(check(C,C1,N))
    {
        printf("true \n");
    }
    else 
    {
        printf("wrong \n");
    }

    cudaEvent_t start1 ,stop1;
    float ms1 = 0;
    dim3 Grid1 (N/THREAD_PER_BLOCK/2,1);
    dim3 Block1(THREAD_PER_BLOCK,1);
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    cudaEventRecord(start1);
    for (int i=0;i<1000;i++)
    {
        elementwise_add_2<<<Grid1,Block1>>>(d_A,d_B,d_C);
    }
    cudaEventRecord(stop1);
    cudaEventSynchronize(stop1);
    cudaEventElapsedTime(&ms1,start1,stop1);
    printf("ms : %.2f \n",ms1/1000);


    cudaMemcpy(C1,d_C,N*sizeof(float),cudaMemcpyDeviceToHost);
    if(check(C,C1,N))
    {
        printf("true \n");
    }
    else 
    {
        printf("wrong \n");
    }    

    free(A);
    free(B);
    free(C);
    free(C1);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

}