#include <cuda.h>
#include <kernels.h>

#define THREADS_PER_BLOCK 64
const int NUM_BLOCKS = N / THREADS_PER_BLOCK + (N % THREADS_PER_BLOCK == 0 ? 0 : 1);
#define GCONST 0.0003f
#define EPS2 0.001f
__device__ float3 v_device[N];
__shared__ float3 shPosition[THREADS_PER_BLOCK];

__device__ float3 force(float3 p2, float3 p, float3 f)
{
	float recrd,recrd3;
	float3 r;
	r.x = p2.x-p.x;
	r.y = p2.y-p.y;
	r.z = p2.z-p.z;
	recrd =  sqrt(r.x*r.x + r.y*r.y + r.z*r.z + EPS2) ;//sqrt(r.x*r.x + r.y*r.y + EPS2);
	recrd3 = recrd * recrd * recrd;
	f.x +=  (r.x * GCONST / recrd3 );
	f.y +=  (r.y * GCONST / recrd3 );
	f.z +=  (r.z * GCONST / recrd3 );
	return f;
}

__device__ float3 accumulate_tile(float3 p, float3 f)
{
	int i;
	for(i=0; i < THREADS_PER_BLOCK; i++)
		f = force(shPosition[i], p, f);
	return f;
}

__global__ void moveParDevice_VBO2(float3 *p_device, float dt)
{
	int k = blockIdx.x * blockDim.x + threadIdx.x;
	if(k>=N) return;
	float3 p, f, v;
	v = v_device[k];
	f = make_float3(0,0,0);
	int i,tile;
	p = p_device[k];
	for(tile=0; tile < N / THREADS_PER_BLOCK; tile++)
	{
		i = tile *  blockDim.x + threadIdx.x;
		shPosition[threadIdx.x] = p_device[i];
		__syncthreads();
		f = accumulate_tile(p, f);
		__syncthreads();
	}
	// integrate
	v.x = v.x + f.x * dt;
	v.y = v.y + f.y * dt;
	v.z = v.z + f.z * dt;
	p.x = p.x + v.x * dt;
	p.y = p.y + v.y * dt;
	p.z = p.z + v.z * dt;
	p_device[k]=p;
	v_device[k]=v;
}

__global__ void moveParDevice_VBO1(float3 *p_device, float dt)
{ 
	int k = blockIdx.x * blockDim.x + threadIdx.x;
	if(k>=N) return;
	// forces
	float3 p, p2, f, v;
	v = v_device[k];
	p = p_device[k];
	f = make_float3(0,0,0);
	for(int i=0; i<N; i++)
	//if(i != k)
	{
		p2 = p_device[i];
		f = force(p2, p, f);
	}
	// integrate
	v.x = v.x + f.x * dt;
	v.y = v.y + f.y * dt;
	v.z = v.z + f.z * dt;
	p.x = p.x + v.x * dt;
	p.y = p.y + v.y * dt;
	p.z = p.z + v.z * dt;
	p_device[k]=p;
	v_device[k]=v;
}

__global__ void moveParDevice_VBO(float3 *p_device, float dt)
{ 
	int k = blockIdx.x * blockDim.x + threadIdx.x;
	if(k>=N) return;
    
	float3 v = v_device[k];
	float3 p = p_device[k];
	p.x = p.x + v.x * dt;
	p.y = p.y + v.y * dt;
	p.z = 50;//p.z + v.z * dt;
	p_device[k]=p;
	v_device[k]=v;
}

void call_movepar_VBO(float3 *points_device, float dt)
{
	moveParDevice_VBO <<< NUM_BLOCKS, THREADS_PER_BLOCK >>> (points_device, dt);
	cudaThreadSynchronize();
}
