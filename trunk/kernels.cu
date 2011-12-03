#include <cuda.h>
#include <iostream>

#include <kernels.h>
#include <operatorsCuda.h>

#define THREADS_PER_BLOCK 64
const int NUM_BLOCKS = N / THREADS_PER_BLOCK + (N % THREADS_PER_BLOCK == 0 ? 0 : 1);
#define GCONST 0.0003f
#define EPS2 0.001f

__device__ float3 v_device[N];
__device__ float3 vNext_device[N];
__device__ float3 pNext_device[N];
__device__ float3 pPrev_device[N];

__device__ void Euler(float3 *p_device, float3 acceleration, int idx, float dt)
{
    vNext_device[idx] = v_device[idx] + (acceleration * dt);
    pNext_device[idx] = p_device[idx] + (vNext_device[idx] * dt);
}

__device__ void calculateForce(float3 *p_device, int idx, float dt)
{
    float mass = 1.f;
    float inversedMass = 1.f / mass;
    float3 force = make_float3(5, 0, 0);
    
    float3 acceleration = inversedMass * force; 
    Euler(p_device, acceleration, idx, dt);
}

__global__ void prepareMove(float3 *p_device, float dt)
{ 
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= N) return;
   
    calculateForce(p_device, idx, dt);
}

__global__ void moveParticle(float3 *p_device, float dt)
{ 
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= N) return;
   
    pPrev_device[idx] = p_device[idx];
    p_device[idx] = pNext_device[idx];
    v_device[idx] = vNext_device[idx];
}

__global__ void particleCollisionWithWalls(float3 *p_device, float dt)
{ 
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= N) return;
    
	float3 v = v_device[idx];
	float3 p = p_device[idx];
	
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
    
    p_device[idx]=p;
	v_device[idx]=v;
}

__global__ void particleCollisionWithOtherParticles(float3 *p_device, float dt)
{

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
    float e = 0.9f;
	
    if(idx >= N || idy >= N) return;

    float nextDistance = length(pNext_device[idx] - pNext_device[idy]);
    float r = 2 * radius;
    float mass = 1.f;
    float reversedMass = 1.f / mass;

    if(nextDistance <= r)
    {
        float3 n = p_device[idx] - p_device[idy];
        normalize(n);

        float reducedMass = 1 / ( 2 * reversedMass);
        float dvn = (v_device[idx] - v_device[idy]) * n;
        float jj = - reducedMass * (e + 1.f) * dvn;

        float3 vx = v_device[idx] + (n * ( jj * reversedMass));
        float3 vy = v_device[idy] - (n * ( jj * reversedMass));

        v_device[idx] = vx;
        v_device[idy] = vy;

        p_device[idx] = p_device[idx] + (n * (r - nextDistance));
        p_device[idy] = p_device[idy] - (n * (r - nextDistance));

        calculateForce(p_device, idx, dt);
        calculateForce(p_device, idy, dt);
    }
}

void call_movepar_VBO(float3 *points_device, float dt)
{
    prepareMove <<< NUM_BLOCKS, THREADS_PER_BLOCK >>> (points_device, dt);
    /*particleCollisionWithOtherParticles <<< NUM_BLOCKS, THREADS_PER_BLOCK >>> (points_device, dt);*/
    moveParticle <<< NUM_BLOCKS, THREADS_PER_BLOCK >>> (points_device, dt);
    particleCollisionWithWalls <<< NUM_BLOCKS, THREADS_PER_BLOCK >>> (points_device, dt);

	cudaThreadSynchronize();
}















/*__device__ float3 force(float3 p2, float3 p, float3 f)*/
/*{*/
	/*float recrd,recrd3;*/
	/*float3 r;*/
	/*r.x = p2.x-p.x;*/
	/*r.y = p2.y-p.y;*/
	/*r.z = p2.z-p.z;*/
	/*recrd =  sqrt(r.x*r.x + r.y*r.y + r.z*r.z + EPS2) ;//sqrt(r.x*r.x + r.y*r.y + EPS2);*/
	/*recrd3 = recrd * recrd * recrd;*/
	/*f.x +=  (r.x * GCONST / recrd3 );*/
	/*f.y +=  (r.y * GCONST / recrd3 );*/
	/*f.z +=  (r.z * GCONST / recrd3 );*/
	/*return f;*/
/*}*/

/*__device__ float3 accumulate_tile(float3 p, float3 f)*/
/*{*/
	/*int i;*/
	/*for(i=0; i < THREADS_PER_BLOCK; i++)*/
		/*f = force(shPosition[i], p, f);*/
	/*return f;*/
/*}*/

/*__global__ void moveParDevice_VBO2(float3 *p_device, float dt)*/
/*{*/
	/*int k = blockIdx.x * blockDim.x + threadIdx.x;*/
	/*if(k>=N) return;*/
	/*float3 p, f, v;*/
	/*v = v_device[k];*/
	/*f = make_float3(0,0,0);*/
	/*int i,tile;*/
	/*p = p_device[k];*/
	/*for(tile=0; tile < N / THREADS_PER_BLOCK; tile++)*/
	/*{*/
		/*i = tile *  blockDim.x + threadIdx.x;*/
		/*shPosition[threadIdx.x] = p_device[i];*/
		/*__syncthreads();*/
		/*f = accumulate_tile(p, f);*/
		/*__syncthreads();*/
	/*}*/
	/*// integrate*/
	/*v.x = v.x + f.x * dt;*/
	/*v.y = v.y + f.y * dt;*/
	/*v.z = v.z + f.z * dt;*/
	/*p.x = p.x + v.x * dt;*/
	/*p.y = p.y + v.y * dt;*/
	/*p.z = p.z + v.z * dt;*/
	/*p_device[k]=p;*/
	/*v_device[k]=v;*/
/*}*/

/*__global__ void moveParDevice_VBO1(float3 *p_device, float dt)*/
/*{ */
	/*int k = blockIdx.x * blockDim.x + threadIdx.x;*/
	/*if(k>=N) return;*/
	/*// forces*/
	/*float3 p, p2, f, v;*/
	/*v = v_device[k];*/
	/*p = p_device[k];*/
	/*f = make_float3(0,0,0);*/
	/*for(int i=0; i<N; i++)*/
	/*//if(i != k)*/
	/*{*/
		/*p2 = p_device[i];*/
		/*f = force(p2, p, f);*/
	/*}*/
	/*// integrate*/
	/*v.x = v.x + f.x * dt;*/
	/*v.y = v.y + f.y * dt;*/
	/*v.z = v.z + f.z * dt;*/
	/*p.x = p.x + v.x * dt;*/
	/*p.y = p.y + v.y * dt;*/
	/*p.z = p.z + v.z * dt;*/
	/*p_device[k]=p;*/
	/*v_device[k]=v;*/
/*}*/
