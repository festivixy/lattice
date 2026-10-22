// Tiled all-pairs gravitational forces with softening, plus leapfrog integration.
#include <cuda_runtime.h>
#include <cstdio>

#define TILE 256
#define SOFTENING2 1e-6f

struct Body {
    float4 pos;  // xyz + mass
    float4 vel;  // xyz + padding
};

__device__ __forceinline__ float3 body_body(float4 bi, float4 bj, float3 acc) {
    float3 r;
    r.x = bj.x - bi.x;
    r.y = bj.y - bi.y;
    r.z = bj.z - bi.z;
    float dist2 = r.x * r.x + r.y * r.y + r.z * r.z + SOFTENING2;
    float inv = rsqrtf(dist2);
    float inv3 = inv * inv * inv;
    float s = bj.w * inv3;
    acc.x += r.x * s;
    acc.y += r.y * s;
    acc.z += r.z * s;
    return acc;
}

__global__ void compute_forces(const float4* __restrict__ pos, float3* __restrict__ acc, int n) {
    __shared__ float4 tile[TILE];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float4 me = i < n ? pos[i] : make_float4(0.f, 0.f, 0.f, 0.f);
    float3 a = make_float3(0.f, 0.f, 0.f);

    for (int base = 0; base < n; base += TILE) {
        int j = base + threadIdx.x;
        tile[threadIdx.x] = j < n ? pos[j] : make_float4(0.f, 0.f, 0.f, 0.f);
        __syncthreads();

        #pragma unroll 16
        for (int k = 0; k < TILE; ++k) {
            a = body_body(me, tile[k], a);
        }
        __syncthreads();
    }
    if (i < n) {
        acc[i] = a;
    }
}

__global__ void kick_drift(float4* pos, float4* vel, const float3* acc, int n, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float4 v = vel[i];
    v.x += 0.5f * dt * acc[i].x;
    v.y += 0.5f * dt * acc[i].y;
    v.z += 0.5f * dt * acc[i].z;
    pos[i].x += dt * v.x;
    pos[i].y += dt * v.y;
    pos[i].z += dt * v.z;
    vel[i] = v;
}

__global__ void kick(float4* vel, const float3* acc, int n, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    vel[i].x += 0.5f * dt * acc[i].x;
    vel[i].y += 0.5f * dt * acc[i].y;
    vel[i].z += 0.5f * dt * acc[i].z;
}

__global__ void kinetic_energy(const float4* pos, const float4* vel, float* out, int n) {
    __shared__ float partial[TILE];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float e = 0.f;
    if (i < n) {
        float4 v = vel[i];
        e = 0.5f * pos[i].w * (v.x * v.x + v.y * v.y + v.z * v.z);
    }
    partial[threadIdx.x] = e;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(out, partial[0]);
    }
}

void step(float4* pos, float4* vel, float3* acc, int n, float dt, cudaStream_t stream) {
    dim3 block(TILE);
    dim3 grid((n + TILE - 1) / TILE);
    kick_drift<<<grid, block, 0, stream>>>(pos, vel, acc, n, dt);
    compute_forces<<<grid, block, 0, stream>>>(pos, acc, n);
    kick<<<grid, block, 0, stream>>>(vel, acc, n, dt);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "step failed: %s\n", cudaGetErrorString(err));
    }
}
