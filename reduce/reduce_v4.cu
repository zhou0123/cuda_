#include<bits/stdc++.h>
#include<cuda.h>
#include"cuda_runtime.h"
#include"device_launch_parameters.h"
#include<time.h>
#include<sys/time.h>
#define THREAD_PER_BLOCK 256

bool check(float *a,float *b,int block_num)
{
    for (int i=0;i<block_num;i++)
    {
        if (a[i]!=b[i]) 
        {
            return false;
        }
    }
    return true;
}
__device__ void warpreduce(volatile float * cache,int tid)
{
    cache[tid]+=cache[tid+32];
    cache[tid]+=cache[tid+16];
    cache[tid]+=cache[tid+8];
    cache[tid]+=cache[tid+4];
    cache[tid]+=cache[tid+2];
    cache[tid]+=cache[tid+1];
}
__global__ void reduce(float* in ,float * out)
{
    __shared__ float shared_memory[THREAD_PER_BLOCK];

    unsigned int tid = threadIdx.x;
    unsigned int id = blockIdx.x*blockDim.x+threadIdx.x;
    shared_memory[tid] = in[id];
    __syncthreads();
    if (tid <128)
    {
        shared_memory[tid] +=shared_memory[tid+128];
    }
    __syncthreads(); 
    if (tid <64)
    {
        shared_memory[tid] +=shared_memory[tid+64];
    }
    __syncthreads(); 
    if (tid <32)
    {
        warpreduce(shared_memory,tid);
    }
    if (tid==0) out[blockIdx.x] = shared_memory[0];
}
int main()
{
    const int N  = 32*1024*1024;
    float * a = (float*)malloc(N*sizeof(float));
    float * d_a ;
    cudaMalloc((void**)&d_a,N*sizeof(float));

    const int block_num = N/THREAD_PER_BLOCK;

    float * out = (float*)malloc(block_num*sizeof(float));
    float * d_out;
    cudaMalloc((void**)&d_out,block_num*sizeof(float));
    float *res  = (float*)malloc(block_num*sizeof(float));
    for (int i=0;i<N;i++) a[i]=1;
    for (int i=0;i<block_num;i++)
    {
        int cur = 0;
        for (int j=0;j<THREAD_PER_BLOCK;j++)
        {
            cur += a[i*THREAD_PER_BLOCK+j];
        }
        res[i] = cur;
    }
    cudaMemcpy(d_a,a,N*sizeof(float),cudaMemcpyHostToDevice);
    dim3 Grid (N/THREAD_PER_BLOCK,1);
    dim3 Block (THREAD_PER_BLOCK,1);
    reduce<<<Grid,Block>>>(d_a,d_out);
    cudaMemcpy(out,d_out,block_num*sizeof(float),cudaMemcpyDeviceToHost);

    if (check(out,res,block_num))
    {
        printf("true");  
    }
    else printf("wrong");
    cudaFree(d_out);
    cudaFree(d_a);
}