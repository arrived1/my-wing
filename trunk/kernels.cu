#include <cuda.h>
#include <iostream>

#include <kernels.h>
#include <operatorsCuda.h>

#define THREADS_PER_BLOCK 64
const int NUM_BLOCKS = N / THREADS_PER_BLOCK + (N % THREADS_PER_BLOCK == 0 ? 0 : 1);
#define GCONST 0.0003f
#define EPS2 0.001f

//__device__ float3 vel[N];
__device__ float3 prevPos[N];

__global__ void copyPreviousPositions(float3 *pos)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= N) return;
    
    prevPos[idx] = pos[idx];
}

__global__ void moveParticle(float3 *pos, float3 *vel, float dt)
{ 
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= N) return;
   
    pos[idx] = pos[idx] + vel[idx] * dt;
    /*vel[idx] = make_float3(0, 0, 0);*/
}

__global__ void particleCollisionWithWalls(float3 *pos, float3 *vel, float dt)
{ 
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= N) return;
    
	float3 v = vel[idx];
	float3 p = pos[idx];
	
    if(p.x < -box/2)
        p.x = box/2;
    if(p.x > box/2)
        p.x = -box/2;
    if(p.y < -box/2)
    {
        p.y = -box - p.y;
        v.y = -v.y;
    }
    if(p.y > box/2)
    {
        p.y = box - p.y;
        v.y = -v.y;
    }
    if(p.z < -box/2)
    {
        p.z = -box - p.z;
        v.z = -v.z;
    }
    if(p.z > box/2)
    {
        p.z = box - p.z;
        v.z = -v.z;
    }
    
    pos[idx]=p;
	vel[idx]=v;
}

__global__ void particleCollisionWithOtherParticles(float3 *pos, float3 *vel, float4 *col, float dt)
{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= N) return;

    float e = 0.9f;
    float mass = 1.f;
    float reversedMass = 1.f / mass;
    
    for(int i = 0; i < N; i++)
    {
        *col = make_float4(0, 0, 0, 0);
        /*float nextDistance = length(prevPos[idx] - prevPos[i]);*/
        if(idx != i)
        {
            float nextDistance = (prevPos[i].x - prevPos[idx].x) *(prevPos[i].x - prevPos[idx].x)+ 
                (prevPos[i].y - prevPos[idx].y)* (prevPos[i].y - prevPos[idx].y) +
                (prevPos[i].z - prevPos[idx].z)* (prevPos[i].z - prevPos[idx].z);

            float r = radius * radius;
            col[idx] = make_float4(pos[idx].x, nextDistance, idx, r);

            if(nextDistance <= r)
            {
                col[idx] = make_float4(pos[idx].x, nextDistance, -69, -69);
                vel[idx] = vel[idx] * (-1);   
                /*float3 n = prevPos[idx] - prevPos[i];*/
                /*normalize(n);*/

                /*float reducedMass = 1 / ( 2 * reversedMass);*/
                /*float dvn = (vel[idx] - vel[i]) * n;*/
                /*float jj = - reducedMass * (e + 1.f) * dvn;*/

                /*vel[idx] = prevPos[idx] + (n * ( jj * reversedMass));*/
                /*pos[idx] = prevPos[idx] + (n * (radius*2 - nextDistance));*/
            }
        }
    }
}

float3 *dvel;
float4 *particleCol;

void call_movepar_VBO(float3 *dpos, float3 *hvel, float dt)
{
    float4 data[N] = { make_float4(0, 0, 0, 0), make_float4(0, 0, 0, 0)};
    cudaMalloc((void**) &particleCol, N * sizeof(float4));
    cudaMalloc((void**) &dvel, N * sizeof(float3));
    cudaMemcpy(dvel, hvel, N * sizeof(float3), cudaMemcpyHostToDevice);
    
    copyPreviousPositions <<< NUM_BLOCKS, THREADS_PER_BLOCK >>> (dpos);
    moveParticle <<< NUM_BLOCKS, THREADS_PER_BLOCK >>> (dpos, dvel, dt);
    particleCollisionWithOtherParticles <<< NUM_BLOCKS, THREADS_PER_BLOCK >>> (dpos, dvel, particleCol, dt);
    cudaThreadSynchronize();
    particleCollisionWithWalls <<< NUM_BLOCKS, THREADS_PER_BLOCK >>> (dpos, dvel, dt);



    cudaMemcpy(data, particleCol, sizeof(float4) * N, cudaMemcpyDeviceToHost);
    static int krok = 0;
    if(data[0].y || data[1].y)
    {
        std::cout << krok++ << ")\t";
        std::cout << "(" << data[0].x << ", " << data[0].y << ", " << data[0].z << " , " << data[0].w <<")\t"
                  << "(" << data[1].x << ", " << data[1].y << ", " << data[1].z << " , " << data[1].w <<")\n";
    }

}
