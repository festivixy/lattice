// Double SHA-256 over a fixed 80-byte header with a precomputed midstate.
#include <cstdint>
#include <cuda_runtime.h>

__constant__ uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

#define ROTR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
#define CH(x, y, z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x, y, z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define S0(x) (ROTR(x, 2) ^ ROTR(x, 13) ^ ROTR(x, 22))
#define S1(x) (ROTR(x, 6) ^ ROTR(x, 11) ^ ROTR(x, 25))
#define s0(x) (ROTR(x, 7) ^ ROTR(x, 18) ^ ((x) >> 3))
#define s1(x) (ROTR(x, 17) ^ ROTR(x, 19) ^ ((x) >> 10))

__device__ void compress(uint32_t state[8], const uint32_t block[16]) {
    uint32_t w[64];
    #pragma unroll
    for (int i = 0; i < 16; ++i) w[i] = block[i];
    #pragma unroll
    for (int i = 16; i < 64; ++i) w[i] = s1(w[i - 2]) + w[i - 7] + s0(w[i - 15]) + w[i - 16];

    uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint32_t e = state[4], f = state[5], g = state[6], h = state[7];
    #pragma unroll
    for (int i = 0; i < 64; ++i) {
        uint32_t t1 = h + S1(e) + CH(e, f, g) + K[i] + w[i];
        uint32_t t2 = S0(a) + MAJ(a, b, c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

__device__ __forceinline__ int leading_zero_bits(const uint32_t digest[8]) {
    int bits = 0;
    for (int i = 7; i >= 0; --i) {
        uint32_t word = __byte_perm(digest[i], 0, 0x0123);
        if (word) return bits + __clz(word);
        bits += 32;
    }
    return bits;
}

__global__ void search(const uint32_t* midstate, const uint32_t* tail, uint64_t base,
                       int target_bits, unsigned long long* found, int* best) {
    uint64_t nonce = base + blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    uint32_t block[16];
    block[0] = tail[0];
    block[1] = tail[1];
    block[2] = tail[2];
    block[3] = (uint32_t)nonce;
    block[4] = 0x80000000;
    #pragma unroll
    for (int i = 5; i < 15; ++i) block[i] = 0;
    block[15] = 640;

    uint32_t state[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) state[i] = midstate[i];
    compress(state, block);

    uint32_t second[16] = {state[0], state[1], state[2], state[3],
                           state[4], state[5], state[6], state[7],
                           0x80000000, 0, 0, 0, 0, 0, 0, 256};
    uint32_t digest[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                          0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    compress(digest, second);

    int zeros = leading_zero_bits(digest);
    atomicMax(best, zeros);
    if (zeros >= target_bits) {
        atomicExch(found, (unsigned long long)nonce);
    }
}
