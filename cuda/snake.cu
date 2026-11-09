// snake.cu - GPU port of lotterSnake (BTC puzzles 71..160).
//
// Idea (same as worker.js, with the bit-alignment bug fixed):
//   V is a random 159-bit number (75..85 ones at start).
//   For puzzle P the candidate key is  key_P = 2^(P-1) + (V mod 2^(P-1)).
//   After all puzzles are tried: V = (V >> 1) | randomBit << 158   ("snake" step).
// GPU trick: key_P - key_(P-1) = 2^(P-2) * (1 + bit_(P-2)(V)), so walking 71 -> 160
// costs one point addition per puzzle; only key_71 needs a (windowed) scalar mult.
//
// Usage: snake [--test] [--seconds N] [--nocool] [--addr path/allPuzzles.js]
#include <cstdio>
#include <cstdarg>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <fstream>
#include <random>
#include <chrono>
#include <thread>
#include <set>
#include <sstream>
#include <tuple>
#include <sys/random.h>
#include <unistd.h>
#include <ctime>
#include <cmath>
#include <algorithm>
#include <termios.h>
#include <sys/ioctl.h>
#include <csignal>
#include <cuda_runtime.h>

typedef uint8_t u8; typedef uint32_t u32; typedef uint64_t u64;
typedef unsigned __int128 u128;
#define HD __host__ __device__ __forceinline__
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)

// Already-solved puzzles inside 71..160 (skipped in search, used as test vectors).
static const int SOLVED[] = {75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135};
#define TITLE_TXT "snake: bitcoin puzzle 71..160"
static const char* BANNER = "\x1b]0;" TITLE_TXT "\x07\x1b[93m" TITLE_TXT "\x1b[0m\n";   // window title + a line on screen

// ---------------- live status line ----------------
// Periodic progress is rewritten in place: "\r" + text + clear-to-end, no newline.
// So the terminal keeps one changing line instead of growing a log every cycle.
// Call status_end() before printing a normal (multi-line) message.
static bool g_status_line = false;
static void tty_size(int* cols, int* rows) {   // real terminal size (stdout may be a pipe into tee)
    *cols = 100; *rows = 24;
    FILE* f = fopen("/dev/tty", "r");
    if (f) {
        struct winsize w = {};
        if (ioctl(fileno(f), TIOCGWINSZ, &w) == 0) {
            if (w.ws_col > 0) *cols = w.ws_col;
            if (w.ws_row > 0) *rows = w.ws_row;
        }
        fclose(f);
    }
}
static void live_write(FILE* f, int cols, const char* buf) {   // one line, kept inside the window width
    if (cols < 20) cols = 20;
    fputc('\r', f);
    int n = (int)strlen(buf);
    fwrite(buf, 1, (size_t)std::min(n, cols - 1), f);   // a wrapped line would break the in-place redraw
    fputs("\x1b[0m\x1b[K", f);
    fflush(f);
}
static void status_printf(const char* fmt, ...) {
    char buf[2048];
    va_list ap; va_start(ap, fmt); vsnprintf(buf, sizeof buf, fmt, ap); va_end(ap);
    int cols, rows; tty_size(&cols, &rows);
    live_write(stdout, cols, buf);
    g_status_line = true;
}
static void status_end() {
    if (!g_status_line) return;
    fputs("\r\x1b[K", stdout);   // wipe the live line, leave the cursor at column 0
    fflush(stdout);
    g_status_line = false;
}

// ---------------- field arithmetic mod p = 2^256 - 2^32 - 977, 8 x 32-bit limbs LE ----------------
struct fe { u32 v[8]; };
struct aff { fe x, y; };
struct jac { fe X, Y, Z; };

HD bool fe_gep(const fe& r) {  // r >= p
    for (int i = 7; i >= 2; i--) if (r.v[i] != 0xFFFFFFFFu) return false;
    return r.v[1] > 0xFFFFFFFEu || (r.v[1] == 0xFFFFFFFEu && r.v[0] >= 0xFFFFFC2Fu);
}
HD void fe_add_c(fe& r) {  // r += 2^256 - p (mod 2^256)
    u64 t = (u64)r.v[0] + 0x3D1u; r.v[0] = (u32)t; t >>= 32;
    t += (u64)r.v[1] + 1; r.v[1] = (u32)t; t >>= 32;
    for (int i = 2; i < 8; i++) { t += r.v[i]; r.v[i] = (u32)t; t >>= 32; }
}
HD void fe_add(fe& r, const fe& a, const fe& b) {
    u64 c = 0;
    for (int i = 0; i < 8; i++) { c += (u64)a.v[i] + b.v[i]; r.v[i] = (u32)c; c >>= 32; }
    if (c || fe_gep(r)) fe_add_c(r);
}
HD void fe_sub(fe& r, const fe& a, const fe& b) {
    u64 br = 0;
    for (int i = 0; i < 8; i++) { u64 t = (u64)a.v[i] - b.v[i] - br; r.v[i] = (u32)t; br = (t >> 32) & 1; }
    if (br) {  // r += p
        const u32 P0 = 0xFFFFFC2Fu, P1 = 0xFFFFFFFEu;
        u64 c = (u64)r.v[0] + P0; r.v[0] = (u32)c; c >>= 32;
        c += (u64)r.v[1] + P1; r.v[1] = (u32)c; c >>= 32;
        for (int i = 2; i < 8; i++) { c += (u64)r.v[i] + 0xFFFFFFFFu; r.v[i] = (u32)c; c >>= 32; }
    }
}
#ifdef __CUDA_ARCH__
// One row of the 256x256 product: t[i..i+8] += a_i * b, as a single carry chain (lo parts, then hi parts).
#define MUL_ROW(i) asm( \
    "mad.lo.cc.u32  %0, %9, %10, %0;\n\t" "madc.lo.cc.u32 %1, %9, %11, %1;\n\t" "madc.lo.cc.u32 %2, %9, %12, %2;\n\t" \
    "madc.lo.cc.u32 %3, %9, %13, %3;\n\t" "madc.lo.cc.u32 %4, %9, %14, %4;\n\t" "madc.lo.cc.u32 %5, %9, %15, %5;\n\t" \
    "madc.lo.cc.u32 %6, %9, %16, %6;\n\t" "madc.lo.cc.u32 %7, %9, %17, %7;\n\t" "addc.u32 %8, 0, 0;\n\t" \
    "mad.hi.cc.u32  %1, %9, %10, %1;\n\t" "madc.hi.cc.u32 %2, %9, %11, %2;\n\t" "madc.hi.cc.u32 %3, %9, %12, %3;\n\t" \
    "madc.hi.cc.u32 %4, %9, %13, %4;\n\t" "madc.hi.cc.u32 %5, %9, %14, %5;\n\t" "madc.hi.cc.u32 %6, %9, %15, %6;\n\t" \
    "madc.hi.cc.u32 %7, %9, %16, %7;\n\t" "madc.hi.u32    %8, %9, %17, %8;" \
    : "+r"(t[i]), "+r"(t[i + 1]), "+r"(t[i + 2]), "+r"(t[i + 3]), "+r"(t[i + 4]), "+r"(t[i + 5]), "+r"(t[i + 6]), "+r"(t[i + 7]), "+r"(t[i + 8]) \
    : "r"(a.v[i]), "r"(b.v[0]), "r"(b.v[1]), "r"(b.v[2]), "r"(b.v[3]), "r"(b.v[4]), "r"(b.v[5]), "r"(b.v[6]), "r"(b.v[7]))
#endif
HD void fe_mul(fe& r, const fe& a, const fe& b) {
    u32 t[16];
    for (int i = 0; i < 16; i++) t[i] = 0;
#ifdef __CUDA_ARCH__
    MUL_ROW(0); MUL_ROW(1); MUL_ROW(2); MUL_ROW(3); MUL_ROW(4); MUL_ROW(5); MUL_ROW(6); MUL_ROW(7);
    // 2^256 == 2^32 + 977 (mod p): r = lo + hi*977 + (hi << 32), 10 words, top two -> c
    u32 r8, r9;
    asm("mad.lo.cc.u32  %0, %18, %26, %10;\n\t" "madc.lo.cc.u32 %1, %19, %26, %11;\n\t" "madc.lo.cc.u32 %2, %20, %26, %12;\n\t"
        "madc.lo.cc.u32 %3, %21, %26, %13;\n\t" "madc.lo.cc.u32 %4, %22, %26, %14;\n\t" "madc.lo.cc.u32 %5, %23, %26, %15;\n\t"
        "madc.lo.cc.u32 %6, %24, %26, %16;\n\t" "madc.lo.cc.u32 %7, %25, %26, %17;\n\t" "addc.u32 %8, 0, 0;\n\t"
        "mad.hi.cc.u32  %1, %18, %26, %1;\n\t" "madc.hi.cc.u32 %2, %19, %26, %2;\n\t" "madc.hi.cc.u32 %3, %20, %26, %3;\n\t"
        "madc.hi.cc.u32 %4, %21, %26, %4;\n\t" "madc.hi.cc.u32 %5, %22, %26, %5;\n\t" "madc.hi.cc.u32 %6, %23, %26, %6;\n\t"
        "madc.hi.cc.u32 %7, %24, %26, %7;\n\t" "madc.hi.u32    %8, %25, %26, %8;\n\t"
        "add.cc.u32  %1, %1, %18;\n\t" "addc.cc.u32 %2, %2, %19;\n\t" "addc.cc.u32 %3, %3, %20;\n\t" "addc.cc.u32 %4, %4, %21;\n\t"
        "addc.cc.u32 %5, %5, %22;\n\t" "addc.cc.u32 %6, %6, %23;\n\t" "addc.cc.u32 %7, %7, %24;\n\t" "addc.cc.u32 %8, %8, %25;\n\t"
        "addc.u32 %9, 0, 0;"
        : "=r"(r.v[0]), "=r"(r.v[1]), "=r"(r.v[2]), "=r"(r.v[3]), "=r"(r.v[4]), "=r"(r.v[5]), "=r"(r.v[6]), "=r"(r.v[7]), "=r"(r8), "=r"(r9)
        : "r"(t[0]), "r"(t[1]), "r"(t[2]), "r"(t[3]), "r"(t[4]), "r"(t[5]), "r"(t[6]), "r"(t[7]),
          "r"(t[8]), "r"(t[9]), "r"(t[10]), "r"(t[11]), "r"(t[12]), "r"(t[13]), "r"(t[14]), "r"(t[15]), "r"(977u));
    u64 c = r8 | ((u64)r9 << 32);     // c < 2^34
#else
    for (int i = 0; i < 8; i++) {
        u64 c = 0;
        for (int j = 0; j < 8; j++) { c += (u64)a.v[i] * b.v[j] + t[i + j]; t[i + j] = (u32)c; c >>= 32; }
        t[i + 8] = (u32)c;
    }
    // 2^256 == 2^32 + 977 (mod p)
    u64 c = 0;
    for (int i = 0; i < 8; i++) {
        c += (u64)t[i] + (u64)t[8 + i] * 977u;
        if (i) c += t[7 + i];
        r.v[i] = (u32)c; c >>= 32;
    }
    c += t[15];                       // c < 2^34
#endif
    u64 m = c * 977u;
    u64 s = (u64)r.v[0] + (m & 0xFFFFFFFFu); r.v[0] = (u32)s; s >>= 32;
    s += (u64)r.v[1] + (m >> 32) + (c & 0xFFFFFFFFu); r.v[1] = (u32)s; s >>= 32;
    s += c >> 32;
    #pragma unroll
    for (int i = 2; i < 8; i++) { s += r.v[i]; r.v[i] = (u32)s; s >>= 32; }
    if (s) fe_add_c(r);
    if (fe_gep(r)) fe_add_c(r);
}
HD void fe_sqr(fe& r, const fe& a) { fe_mul(r, a, a); }
HD void fe_sqrn(fe& r, const fe& a, int n) { r = a; for (int i = 0; i < n; i++) fe_mul(r, r, r); }
HD void fe_inv(fe& r, const fe& a) {  // a^(p-2), libsecp256k1 addition chain
    fe x2, x3, x6, x9, x11, x22, x44, x88, x176, x220, x223, t;
    fe_sqr(x2, a); fe_mul(x2, x2, a);
    fe_sqr(x3, x2); fe_mul(x3, x3, a);
    fe_sqrn(x6, x3, 3); fe_mul(x6, x6, x3);
    fe_sqrn(x9, x6, 3); fe_mul(x9, x9, x3);
    fe_sqrn(x11, x9, 2); fe_mul(x11, x11, x2);
    fe_sqrn(x22, x11, 11); fe_mul(x22, x22, x11);
    fe_sqrn(x44, x22, 22); fe_mul(x44, x44, x22);
    fe_sqrn(x88, x44, 44); fe_mul(x88, x88, x44);
    fe_sqrn(x176, x88, 88); fe_mul(x176, x176, x88);
    fe_sqrn(x220, x176, 44); fe_mul(x220, x220, x44);
    fe_sqrn(x223, x220, 3); fe_mul(x223, x223, x3);
    fe_sqrn(t, x223, 23); fe_mul(t, t, x22);
    fe_sqrn(t, t, 5); fe_mul(t, t, a);
    fe_sqrn(t, t, 3); fe_mul(t, t, x2);
    fe_sqrn(t, t, 2); fe_mul(r, t, a);
}

// P += Q (Q affine). ponytail: no P==Q / P==-Q / infinity cases - impossible for keys < 2^160 here (needs V mod 2^k == 0).
HD void madd(jac& P, const aff& Q) {
    fe z1z1, u2, s2, h, r, hh, hhh, v, t;
    fe_sqr(z1z1, P.Z); fe_mul(u2, Q.x, z1z1);
    fe_mul(s2, Q.y, P.Z); fe_mul(s2, s2, z1z1);
    fe_sub(h, u2, P.X); fe_sub(r, s2, P.Y);
    fe_sqr(hh, h); fe_mul(hhh, h, hh); fe_mul(v, P.X, hh);
    fe_sqr(t, r); fe_sub(t, t, hhh); fe_sub(t, t, v); fe_sub(t, t, v);
    fe_mul(P.Z, P.Z, h);
    fe_sub(v, v, t); fe_mul(v, v, r); fe_mul(hhh, hhh, P.Y); fe_sub(P.Y, v, hhh);
    P.X = t;
}

// ---------------- SHA-256 / RIPEMD-160 (single block) ----------------
#define SHA_K 0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,\
 0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,\
 0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,\
 0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,\
 0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,\
 0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,\
 0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,\
 0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
__constant__ u32 dSHA_K[64] = {SHA_K};
[[maybe_unused]] static const u32 hSHA_K[64] = {SHA_K};
#ifdef __CUDA_ARCH__
#define SK dSHA_K
#else
#define SK hSHA_K
#endif
HD u32 ror(u32 x, int n) { return (x >> n) | (x << (32 - n)); }
HD u32 rol(u32 x, int n) { return (x << n) | (x >> (32 - n)); }
HD u32 bswap(u32 x) { return (x >> 24) | ((x >> 8) & 0xFF00u) | ((x << 8) & 0xFF0000u) | (x << 24); }

HD void sha256_init(u32 s[8]) {
    s[0] = 0x6a09e667; s[1] = 0xbb67ae85; s[2] = 0x3c6ef372; s[3] = 0xa54ff53a;
    s[4] = 0x510e527f; s[5] = 0x9b05688c; s[6] = 0x1f83d9ab; s[7] = 0x5be0cd19;
}
HD void sha256_block(u32 st[8], const u32 in[16]) {
    u32 w[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) w[i] = in[i];
    u32 a = st[0], b = st[1], c = st[2], d = st[3], e = st[4], f = st[5], g = st[6], h = st[7];
    #pragma unroll
    for (int i = 0; i < 64; i++) {
        if (i >= 16) {
            u32 x = w[(i - 15) & 15], y = w[(i - 2) & 15];
            w[i & 15] += (ror(y, 17) ^ ror(y, 19) ^ (y >> 10)) + w[(i - 7) & 15] + (ror(x, 7) ^ ror(x, 18) ^ (x >> 3));
        }
        u32 t1 = h + (ror(e, 6) ^ ror(e, 11) ^ ror(e, 25)) + ((e & f) ^ (~e & g)) + SK[i] + w[i & 15];
        u32 t2 = (ror(a, 2) ^ ror(a, 13) ^ ror(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
        h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    st[0] += a; st[1] += b; st[2] += c; st[3] += d; st[4] += e; st[5] += f; st[6] += g; st[7] += h;
}

#define F1(x, y, z) ((x) ^ (y) ^ (z))
#define F2(x, y, z) (((x) & (y)) | (~(x) & (z)))
#define F3(x, y, z) (((x) | ~(y)) ^ (z))
#define F4(x, y, z) (((x) & (z)) | ((y) & ~(z)))
#define F5(x, y, z) ((x) ^ ((y) | ~(z)))
#define RMD_STEP(a, b, c, d, e, F, x, k, s) { a = rol(a + F(b, c, d) + (x) + (k), s) + e; c = rol(c, 10); }
HD void rmd160_init(u32 h[5]) { h[0] = 0x67452301; h[1] = 0xEFCDAB89; h[2] = 0x98BADCFE; h[3] = 0x10325476; h[4] = 0xC3D2E1F0; }
HD void rmd160_block(u32 h[5], const u32 X[16]) {
    u32 a1 = h[0], b1 = h[1], c1 = h[2], d1 = h[3], e1 = h[4];
    u32 a2 = a1, b2 = b1, c2 = c1, d2 = d1, e2 = e1;
#include "rmd160_steps.h"
    u32 t = h[1] + c1 + d2;
    h[1] = h[2] + d1 + e2; h[2] = h[3] + e1 + a2; h[3] = h[4] + a1 + b2; h[4] = h[0] + b1 + c2; h[0] = t;
}

// hash160 of compressed pubkey (x, parity). Output words = RIPEMD state (digest bytes little-endian).
HD void hash160_pub(const fe& x, u32 parity, u32 out[5]) {
    u32 w[16];
    w[0] = ((2u | parity) << 24) | (x.v[7] >> 8);
    #pragma unroll
    for (int i = 1; i < 8; i++) w[i] = (x.v[8 - i] << 24) | (x.v[7 - i] >> 8);
    w[8] = (x.v[0] << 24) | 0x800000u;
    #pragma unroll
    for (int i = 9; i < 15; i++) w[i] = 0;
    w[15] = 33 * 8;
    u32 st[8]; sha256_init(st); sha256_block(st, w);
    u32 m[16];
    #pragma unroll
    for (int i = 0; i < 8; i++) m[i] = bswap(st[i]);
    m[8] = 0x80; m[9] = m[10] = m[11] = m[12] = m[13] = 0; m[14] = 256; m[15] = 0;
    rmd160_init(out); rmd160_block(out, m);
}

// ===== GEN_SCAN_CORE_END =====

// ---------------- GPU search kernel ----------------
#define CHUNK 16
struct Hit { u32 V[5]; u32 p, step, tid; };
__constant__ u32 c_tgt[161][5];
__constant__ u8 c_active[161];
__constant__ aff c_pow2[161];          // 2^k * G
__device__ aff* g_win;                 // [9][256]: j * 2^(8w) * G  (window table for key_71)

__device__ void flush(fe* X, fe* Y, fe* Z, fe* acc, const u8* ps, int n, const u32* V, int vs, u32 step, u32 tid, u32* hitN, Hit* hits) {
    acc[0] = Z[0];
    for (int i = 1; i < n; i++) fe_mul(acc[i], acc[i - 1], Z[i]);
    fe inv; fe_inv(inv, acc[n - 1]);
    for (int i = n - 1; i >= 0; i--) {
        fe zi, zi2, x, y;
        if (i) { fe_mul(zi, inv, acc[i - 1]); fe_mul(inv, inv, Z[i]); } else zi = inv;
        fe_sqr(zi2, zi); fe_mul(x, X[i], zi2); fe_mul(y, Y[i], zi2); fe_mul(y, y, zi);
        u32 h[5]; hash160_pub(x, y.v[0] & 1, h);
        int p = ps[i];
        if (h[0] == c_tgt[p][0] && h[1] == c_tgt[p][1] && h[2] == c_tgt[p][2] && h[3] == c_tgt[p][3] && h[4] == c_tgt[p][4]) {
            u32 k = atomicAdd(hitN, 1);
            if (k < 1024) { for (int j = 0; j < 5; j++) hits[k].V[j] = V[i * vs + j]; hits[k].p = p; hits[k].step = step; hits[k].tid = tid; }
        }
    }
}

__global__ void __launch_bounds__(256) k_snake(u32* gV, u64* gR, int steps, u32* hitN, Hit* hits) {
    u32 tid = blockIdx.x * blockDim.x + threadIdx.x;
    u32 V[5]; for (int i = 0; i < 5; i++) V[i] = gV[tid * 5 + i];
    u64 rs = gR[tid];
    fe X[CHUNK], Y[CHUNK], Z[CHUNK], acc[CHUNK]; u8 ps[CHUNK];
    for (int s = 0; s < steps; s++) {
        // key_71 = 2^70 | (V mod 2^70), windows of 8 bits
        jac P;
        const aff& top = g_win[8 * 256 + ((V[2] & 0x3F) | 0x40)];
        P.X = top.x; P.Y = top.y; P.Z = fe{{1, 0, 0, 0, 0, 0, 0, 0}};
        for (int w = 7; w >= 0; w--) {
            u32 j = (V[w >> 2] >> (8 * (w & 3))) & 0xFF;
            if (j) madd(P, g_win[w * 256 + j]);
        }
        int n = 0;
        for (int p = 71; p <= 160; p++) {
            if (p > 71) { int k = p - 2; madd(P, c_pow2[k + ((V[k >> 5] >> (k & 31)) & 1)]); }
            if (c_active[p]) {
                X[n] = P.X; Y[n] = P.Y; Z[n] = P.Z; ps[n] = (u8)p;
                if (++n == CHUNK) { flush(X, Y, Z, acc, ps, n, V, 0, s, tid, hitN, hits); n = 0; }
            }
        }
        if (n) flush(X, Y, Z, acc, ps, n, V, 0, s, tid, hitN, hits);
        // snake step: V = V >> 1 | randomBit << 158
        rs ^= rs >> 12; rs ^= rs << 25; rs ^= rs >> 27;
        u32 rb = (u32)((rs * 0x2545F4914F6CDD1DULL) >> 63);
        for (int i = 0; i < 4; i++) V[i] = (V[i] >> 1) | (V[i + 1] << 31);
        V[4] = (V[4] >> 1) | (rb << 30);
    }
    for (int i = 0; i < 5; i++) gV[tid * 5 + i] = V[i];
    gR[tid] = rs;
}

// --spread P: one puzzle only. Key = top bit + (k-1) more ones at random places among the low P-1 bits.
// The ones range --kmin..--kmax is split across threads: thread t always keeps k = kmin + t % (kmax - kmin + 1).
// Each step moves one random 1 to a random 0 place: two point adds (-2^i*G, +2^j*G), no scalar mult.
__device__ __forceinline__ u32 xs32(u64& s) { s ^= s >> 12; s ^= s << 25; s ^= s >> 27; return (u32)((s * 0x2545F4914F6CDD1DULL) >> 32); }
__device__ __forceinline__ void kflip(jac& P, u32* K, int b) {   // K ^= 2^b, P stays K*G
    aff q = c_pow2[b];
    if ((K[b >> 5] >> (b & 31)) & 1) { fe z = {}; fe_sub(q.y, z, q.y); }   // bit set: subtract
    madd(P, q); K[b >> 5] ^= 1u << (b & 31);
}
__device__ __forceinline__ int kpick(u64& rs, const u32* K, int nb, int want) {   // random bit < nb with value `want`
    for (;;) { int b = xs32(rs) % nb; if ((int)((K[b >> 5] >> (b & 31)) & 1) == want) return b; }
}
__global__ void __launch_bounds__(256) k_spread(int p, int kmin, int nk, u32* gK, jac* gP, u64* gR, int steps, u32* hitN, Hit* hits) {
    u32 tid = blockIdx.x * blockDim.x + threadIdx.x;
    int k = kmin + (int)(tid % nk);   // this thread's fixed ones count
    u32 K[5]; for (int i = 0; i < 5; i++) K[i] = gK[tid * 5 + i];
    jac P = gP[tid]; u64 rs = gR[tid];
    int nb = p - 1, ones = 0; for (int i = 0; i < 5; i++) ones += __popc(K[i]);
    while (ones < k) { kflip(P, K, kpick(rs, K, nb, 0)); ones++; }
    while (ones > k) { kflip(P, K, kpick(rs, K, nb, 1)); ones--; }
    fe X[CHUNK], Y[CHUNK], Z[CHUNK], acc[CHUNK]; u8 ps[CHUNK]; u32 KC[CHUNK * 5];
    int n = 0;
    for (int s = 0; s < steps; s++) {
        { int i = kpick(rs, K, nb, 1), j = kpick(rs, K, nb, 0); kflip(P, K, i); kflip(P, K, j); }   // pick both first: every step is a new key
        X[n] = P.X; Y[n] = P.Y; Z[n] = P.Z; ps[n] = (u8)p; for (int w = 0; w < 5; w++) KC[n * 5 + w] = K[w];
        if (++n == CHUNK) { flush(X, Y, Z, acc, ps, n, KC, 5, s, tid, hitN, hits); n = 0; }
    }
    if (n) flush(X, Y, Z, acc, ps, n, KC, 5, steps, tid, hitN, hits);
    for (int i = 0; i < 5; i++) gK[tid * 5 + i] = K[i];
    gP[tid] = P; gR[tid] = rs;
}

// ---------------- --exhaust: deterministic, gap-free walk over one whole fixed-popcount slice ----------------
// A puzzle-p key is bit (p-1) plus w = k-1 more ones among the p-1 low bits.
// For w > (p-1)/2 we enumerate the zeros instead and use key = (2^p - 1) - zeros,
// so the enumerated masks always carry z = min(w, p-1-w) bits (z = 0..9 for the cheap ends).
// Rank <-> mask is the ascending numeric order of fixed-popcount masks (Gosper's hack), so any
// chunk [r0, r1) can be started directly from its rank: no gap, no repeat, no random walk.
__constant__ u64 c_bin[129][36];  // C(n,k) for n<=128, k<=35; saturated at UINT64_MAX
__device__ __forceinline__ u128 bit128(int b) { return b < 64 ? (u128)(1ull << b) : ((u128)(1ull << (b - 64)) << 64); }
__device__ __forceinline__ int get128(u128 x, int b) { return b < 64 ? (int)(((u64)x >> b) & 1) : (int)(((u64)(x >> 64) >> (b - 64)) & 1); }
__device__ __forceinline__ int ctz128(u128 x) {
    u64 lo = (u64)x;
    return lo ? __ffsll((long long)lo) - 1 : 64 + __ffsll((long long)(u64)(x >> 64)) - 1;
}
__device__ __forceinline__ u128 shr128(u128 x, int t) {   // t in [0,127]; avoids __int128 variable shifts
    if (t <= 0) return x;
    u64 lo = (u64)x, hi = (u64)(x >> 64);
    if (t >= 64) return (u128)(hi >> (t - 64));
    return ((u128)(hi >> t) << 64) | ((lo >> t) | (hi << (64 - t)));
}
__device__ __forceinline__ u128 gosper(u128 x) {   // next mask with the same popcount, ascending
    int t = ctz128(x); u128 r = x + bit128(t);
    return r | shr128(shr128(r ^ x, 2), t);
}
__device__ __forceinline__ u128 comb_unrank(u64 r, int nb, int z) {   // z bits among bits 0..nb-1
    u128 M = 0; int k = z;
    while (k > 0) {
        int q = k - 1; u64 c = c_bin[q][k - 1];
        while (r >= c && q < nb - 1) { r -= c; q++; c = c_bin[q][k - 1]; }
        M |= bit128(q); k--;
    }
    return M;
}
__device__ void flush_ex(fe* X, fe* Y, fe* Z, fe* acc, const u128* M, int n, int p, u32 tid, u32* hitN, Hit* hits) {
    acc[0] = Z[0];
    for (int i = 1; i < n; i++) fe_mul(acc[i], acc[i - 1], Z[i]);
    fe inv; fe_inv(inv, acc[n - 1]);
    for (int i = n - 1; i >= 0; i--) {
        fe zi, zi2, x, y;
        if (i) { fe_mul(zi, inv, acc[i - 1]); fe_mul(inv, inv, Z[i]); } else zi = inv;
        fe_sqr(zi2, zi); fe_mul(x, X[i], zi2); fe_mul(y, Y[i], zi2); fe_mul(y, y, zi);
        u32 h[5]; hash160_pub(x, y.v[0] & 1, h);
        if (h[0] == c_tgt[p][0] && h[1] == c_tgt[p][1] && h[2] == c_tgt[p][2] && h[3] == c_tgt[p][3] && h[4] == c_tgt[p][4]) {
            u32 k = atomicAdd(hitN, 1);
            if (k < 1024) {   // V = enumerated mask (lo 64 bits, then hi 64 bits)
                u64 lo = (u64)M[i], hi = (u64)(M[i] >> 64);
                hits[k].V[0] = (u32)lo; hits[k].V[1] = (u32)(lo >> 32);
                hits[k].V[2] = (u32)hi; hits[k].V[3] = (u32)(hi >> 32); hits[k].V[4] = 0;
                hits[k].p = (u32)p; hits[k].step = 1; hits[k].tid = tid;
            }
        }
    }
}
__global__ void __launch_bounds__(256) k_exhaust(int p, int nb, int z, u64 firstChunk, u64 nChunks, u64 chunkSize, u64 total,
                                                 jac base, int invert, u32* hitN, Hit* hits) {
    u64 idx = (u64)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= nChunks) return;
    u64 r0 = (firstChunk + idx) * chunkSize;
    if (r0 >= total) return;
    u64 r1 = r0 + chunkSize; if (r1 > total) r1 = total;
    u128 M = comb_unrank(r0, nb, z);
    jac P = base;                                   // base = 2^(p-1)*G, or (2^p-1)*G when we enumerate zeros
    for (int b = 0; b < nb; b++) if (get128(M, b)) {
        aff q = c_pow2[b]; if (invert) { fe zz = {}; fe_sub(q.y, zz, q.y); } madd(P, q);
    }
    fe X[CHUNK], Y[CHUNK], Z[CHUNK], acc[CHUNK]; u128 MQ[CHUNK];
    int n = 0;
    for (u64 i = r0;; i++) {
        X[n] = P.X; Y[n] = P.Y; Z[n] = P.Z; MQ[n] = M;
        if (++n == CHUNK) { flush_ex(X, Y, Z, acc, MQ, n, p, (u32)idx, hitN, hits); n = 0; }
        if (i + 1 >= r1) break;
        u128 N2 = gosper(M), d = M ^ N2;
        while (d) {                                  // only the changed bits cost a point add
            int b = ctz128(d); u128 bit = bit128(b); d ^= bit;
            aff q = c_pow2[b];
            // add-mode: a bit that leaves the mask is subtracted; zero-mode: it is added (key = base - mask)
            if ((get128(M, b) != 0) != (invert != 0)) { fe zz = {}; fe_sub(q.y, zz, q.y); }
            madd(P, q);
        }
        M = N2;
    }
    if (n) flush_ex(X, Y, Z, acc, MQ, n, p, (u32)idx, hitN, hits);
}

// ---------------- --freeze: m ones at chosen places, every other low bit swept ----------------
// Mask = frozen | s, s running over all 2^f subsets of the free places in ascending rank order:
// next = (s - free) & free. Chunk c starts at scatter(c*CS), so any chunk is reachable by rank.
// Frozen masks overlap (a key with q ones sits in C(q,m) of them), so finished masks are remembered
// in freeze_masks.txt. Slices k<=11 and k>=61 are already exhausted: their keys are not hashed here.
__device__ __forceinline__ u128 scatter128(u64 r, u128 freeMask) {   // rank bits -> free places, ascending
    u128 M = 0, f = freeMask;
    while (f) { int t = ctz128(f); f ^= bit128(t); if (r & 1) M |= bit128(t); r >>= 1; }
    return M;
}
__global__ void __launch_bounds__(256) k_freeze(int p, u64 blo, u64 bhi, u64 flo, u64 fhi, u64 firstChunk, u64 nChunks, u64 chunkSize,
                                                u64 total, jac base, int skipDone, u32* hitN, Hit* hits) {
    u64 idx = (u64)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= nChunks) return;
    u64 r0 = (firstChunk + idx) * chunkSize;
    if (r0 >= total) return;
    u64 r1 = r0 + chunkSize; if (r1 > total) r1 = total;
    u128 B = ((u128)bhi << 64) | blo, F = ((u128)fhi << 64) | flo;
    u128 s = scatter128(r0, F), M = B | s;
    jac P = base;                                   // base = 2^(p-1)*G; the mask bits are added here
    for (u128 d = M; d; ) { int b = ctz128(d); d ^= bit128(b); madd(P, c_pow2[b]); }
    fe X[CHUNK], Y[CHUNK], Z[CHUNK], acc[CHUNK]; u128 MQ[CHUNK];
    int n = 0;
    for (u64 i = r0;; i++) {
        int k = 1 + __popcll((u64)M) + __popcll((u64)(M >> 64));      // ones in the whole key (bit p-1 is set)
        if (!skipDone || (k > 11 && k < 61)) {                        // shortcut: 11/61 = ends already exhausted; parameterise for another puzzle
            X[n] = P.X; Y[n] = P.Y; Z[n] = P.Z; MQ[n] = M;
            if (++n == CHUNK) { flush_ex(X, Y, Z, acc, MQ, n, p, (u32)idx, hitN, hits); n = 0; }
        }
        if (i + 1 >= r1) break;
        u128 s2 = (s - F) & F, d = s ^ s2;                            // only free places change
        s = s2; M = B | s;
        while (d) { int b = ctz128(d); d ^= bit128(b); aff q = c_pow2[b]; if (!get128(M, b)) { fe zz = {}; fe_sub(q.y, zz, q.y); } madd(P, q); }
    }
    if (n) flush_ex(X, Y, Z, acc, MQ, n, p, (u32)idx, hitN, hits);
}

__global__ void k_mul_test(const fe* a, const fe* b, fe* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) fe_mul(out[i], a[i], b[i]);
}

// ===== GEN_SCAN_HOST_BEGIN =====
// ---------------- host helpers ----------------
static fe fe_hex(const char* h) {  // big-endian hex, <= 64 digits
    fe r{}; int n = strlen(h);
    for (int i = 0; i < n; i++) {
        char c = h[n - 1 - i];
        u32 d = c <= '9' ? c - '0' : (c | 32) - 'a' + 10;
        r.v[i / 8] |= d << (4 * (i % 8));
    }
    return r;
}
static std::string fe_str(const fe& a) { char b[65]; for (int i = 0; i < 8; i++) sprintf(b + 8 * i, "%08x", a.v[7 - i]); return b; }
static HD bool fe_eq(const fe& a, const fe& b) {
    u32 d = 0;
    for (int i = 0; i < 8; i++) d |= a.v[i] ^ b.v[i];
    return d == 0;
}
static aff to_aff(const jac& P) {
    fe zi, zi2; fe_inv(zi, P.Z); fe_sqr(zi2, zi);
    aff r; fe_mul(r.x, P.X, zi2); fe_mul(r.y, P.Y, zi2); fe_mul(r.y, r.y, zi); return r;
}
static jac to_jac(const aff& a) { return jac{a.x, a.y, fe{{1, 0, 0, 0, 0, 0, 0, 0}}}; }
static HD void jdbl(jac& P) {
    fe A, B, C, D, E, F, t;
    fe_mul(P.Z, P.Z, P.Y); fe_add(P.Z, P.Z, P.Z);
    fe_sqr(A, P.X); fe_sqr(B, P.Y); fe_sqr(C, B);
    fe_add(t, P.X, B); fe_sqr(t, t); fe_sub(t, t, A); fe_sub(t, t, C); fe_add(D, t, t);
    fe_add(E, A, A); fe_add(E, E, A); fe_sqr(F, E);
    fe_sub(P.X, F, D); fe_sub(P.X, P.X, D);
    fe_sub(t, D, P.X); fe_mul(t, E, t);
    fe_add(C, C, C); fe_add(C, C, C); fe_add(C, C, C); fe_sub(P.Y, t, C);
}
static aff aff_add(const aff& a, const aff& b) {
    jac P = to_jac(a);
    if (fe_eq(a.x, b.x)) jdbl(P); else madd(P, b);
    return to_aff(P);
}
static aff hpow2[161];
static void key_h160(const u32 key[8], u32 h[5]) {  // key < 2^161
    jac acc; bool init = false;
    for (int i = 0; i <= 160; i++) if ((key[i >> 5] >> (i & 31)) & 1) {
        if (!init) { acc = to_jac(hpow2[i]); init = true; } else madd(acc, hpow2[i]);
    }
    aff a = to_aff(acc); hash160_pub(a.x, a.y.v[0] & 1, h);
}
static void sha256_short(const u8* m, int len, u8 out[32]) {  // len <= 55
    u8 b[64] = {0}; memcpy(b, m, len); b[len] = 0x80; b[62] = (len * 8) >> 8; b[63] = (len * 8) & 255;
    u32 w[16]; for (int i = 0; i < 16; i++) w[i] = (b[4 * i] << 24) | (b[4 * i + 1] << 16) | (b[4 * i + 2] << 8) | b[4 * i + 3];
    u32 st[8]; sha256_init(st); sha256_block(st, w);
    for (int i = 0; i < 32; i++) out[i] = st[i / 4] >> (24 - 8 * (i % 4));
}
static void rmd160_short(const u8* m, int len, u8 out[20]) {  // len <= 55
    u8 b[64] = {0}; memcpy(b, m, len); b[len] = 0x80; b[56] = (len * 8) & 255; b[57] = (len * 8) >> 8;
    u32 w[16]; for (int i = 0; i < 16; i++) w[i] = b[4 * i] | (b[4 * i + 1] << 8) | (b[4 * i + 2] << 16) | (b[4 * i + 3] << 24);
    u32 h[5]; rmd160_init(h); rmd160_block(h, w);
    for (int i = 0; i < 20; i++) out[i] = h[i / 4] >> (8 * (i % 4));
}
static std::string hexs(const u8* b, int n) { std::string s; char t[3]; for (int i = 0; i < n; i++) { sprintf(t, "%02x", b[i]); s += t; } return s; }
static bool b58_h160(const std::string& s, u32 h[5]) {  // P2PKH address -> hash160 words; checks checksum
    static const char* A = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
    u8 b[25] = {0};
    for (char ch : s) {
        const char* q = strchr(A, ch); if (!q || !ch) return false;
        u32 c = q - A;
        for (int i = 24; i >= 0; i--) { c += 58u * b[i]; b[i] = c & 255; c >>= 8; }
        if (c) return false;
    }
    u8 d1[32], d2[32]; sha256_short(b, 21, d1); sha256_short(d1, 32, d2);
    if (memcmp(d2, b + 21, 4) || b[0] != 0) return false;
    for (int i = 0; i < 5; i++) h[i] = b[1 + 4 * i] | (b[2 + 4 * i] << 8) | (b[3 + 4 * i] << 16) | ((u32)b[4 + 4 * i] << 24);
    return true;
}
static void key_hex(const char* hex, u32 k[8]) { fe f = fe_hex(hex); memcpy(k, f.v, 32); }
static std::string key_str(const u32 k[8]) { fe f; memcpy(f.v, k, 32); return fe_str(f); }
[[maybe_unused]] static jac point_of_key(const u32 key[8]) {   // sum of 2^b*G over the set bits (slow reference)
    jac acc{}; bool init = false;
    for (int i = 0; i <= 160; i++) if ((key[i >> 5] >> (i & 31)) & 1) {
        if (!init) { acc = to_jac(hpow2[i]); init = true; } else madd(acc, hpow2[i]);
    }
    return acc;
}
// ===== GEN_SCAN_HOST_END =====
static u128 bit128h(int b) { return b < 64 ? (u128)(1ull << b) : ((u128)(1ull << (b - 64)) << 64); }
static u128 scatter_h(u64 r, u128 freeMask, int nb) {   // host twin of scatter128: rank bits -> free places, ascending
    u128 M = 0;
    for (int b = 0; b < nb; b++) if (freeMask & bit128h(b)) { if (r & 1) M |= bit128h(b); r >>= 1; }
    return M;
}
static u128 binom128(int n, int k) {   // exact C(n,k), n <= 128
    if (k < 0 || k > n) return 0;
    u128 r = 1;
    for (int i = 1; i <= k; i++) r = r * (u128)(unsigned)(n - i + 1) / (u128)(unsigned)i;
    return r;
}
static std::string u128_str(u128 v) {   // decimal with thousands separators
    if (!v) return "0";
    std::string s;
    while (v) { s += (char)('0' + (int)(unsigned)(v % 10)); v /= 10; }
    std::reverse(s.begin(), s.end());
    for (int i = (int)s.size() - 3; i > 0; i -= 3) s.insert(i, ",");
    return s;
}
static std::string b58enc(const u8* b, int n) {
    static const char* A = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
    std::vector<u8> d;   // base-58 digits, least significant first
    for (int i = 0; i < n; i++) {
        u32 c = b[i];
        for (auto& x : d) { c += x * 256u; x = c % 58; c /= 58; }
        while (c) { d.push_back(c % 58); c /= 58; }
    }
    std::string r; for (int i = 0; i < n && !b[i]; i++) r += '1';
    for (auto it = d.rbegin(); it != d.rend(); ++it) r += A[*it];
    return r;
}
static std::string wif(const u32 k[8]) {  // compressed-pubkey WIF (the puzzle addresses are compressed)
    u8 b[38], d1[32], d2[32]; b[0] = 0x80;
    for (int i = 0; i < 32; i++) b[1 + i] = k[7 - i / 4] >> (24 - 8 * (i % 4));
    b[33] = 1; sha256_short(b, 34, d1); sha256_short(d1, 32, d2); memcpy(b + 34, d2, 4);
    return b58enc(b, 38);
}
static std::string found_record(int p, const u32 k[8], const std::string& address, bool ok) {
    std::string bin; for (int i = p - 1; i >= 0; i--) bin += (char)('0' + ((k[i >> 5] >> (i & 31)) & 1));
    char when[64]; time_t t = time(nullptr); strftime(when, sizeof when, "%Y-%m-%d %H:%M:%S", localtime(&t));
    return "puzzle   " + std::to_string(p) + "\naddress  " + address +
           "\nverified " + (ok ? "yes (address recomputed from the key on CPU)" : "NO - CPU re-check failed") +
           "\nhex      " + key_str(k) + "\nbinary   " + bin + "\nWIF      " + wif(k) + "  (compressed)\nfound    " + when + "\n\n";
}
static int getbit(const u32* a, int i) { return (a[i >> 5] >> (i & 31)) & 1; }
static void setbit(u32* a, int i, int b) { if (b) a[i >> 5] |= 1u << (i & 31); else a[i >> 5] &= ~(1u << (i & 31)); }
static void V_to_key(const u32 V[5], int p, u32 k[8]) {
    memset(k, 0, 32);
    for (int i = 0; i < p - 1; i++) setbit(k, i, getbit(V, i));
    setbit(k, p - 1, 1);
}
static std::mt19937_64 rng;
static void random_V(u32 V[5]) {  // like generateBinaryString(159, 80, 5)
    memset(V, 0, 20);
    int ones = 75 + rng() % 11;
    for (int c = 0; c < ones;) { int i = rng() % 159; if (!getbit(V, i)) { setbit(V, i, 1); c++; } }
}
// Keyboard control of the running search (single keys, no Enter).
static struct termios kb_old; static bool kb_raw = false;

static void kb_restore() { if (kb_raw) { tcsetattr(0, TCSANOW, &kb_old); kb_raw = false; } }
static void kb_init() {
    if (!isatty(0) || tcgetattr(0, &kb_old)) return;
    struct termios t = kb_old; t.c_lflag &= ~(ICANON | ECHO); t.c_cc[VMIN] = 0; t.c_cc[VTIME] = 0;
    if (tcsetattr(0, TCSANOW, &t)) return;
    kb_raw = true; atexit(kb_restore);
    signal(SIGINT, [](int) { kb_restore(); _exit(130); });
}
static int kb_get() { unsigned char c; return kb_raw && read(0, &c, 1) == 1 ? c : -1; }
static int gpu_temp() {
    FILE* f = popen("Q=\"--query-gpu=temperature.gpu --format=csv,noheader\"; nvidia-smi $Q 2>/dev/null || /usr/lib/wsl/lib/nvidia-smi $Q 2>/dev/null", "r");
    int t = -1; if (f) { if (fscanf(f, "%d", &t) != 1) t = -1; pclose(f); } return t;
}

// Found key: GPU is already stopped; long sound + email via alert.py (SMTP settings in mail.env).
static void alarm(int p, const std::string& address, bool test) {
    kb_restore();
    printf("\a\nSearch stopped. Sound alarm + email to you...\n"); fflush(stdout);
    // the green retro font is hard to read: show the record in Notepad, normal font
    // открыть found.txt в системном просмотрщике (Linux: xdg-open; в WSL/Windows — notepad)
    if (!test && system("(xdg-open found.txt >/dev/null 2>&1 || notepad.exe found.txt >/dev/null 2>&1) &") != 0)
        printf("found.txt written; open it manually\n");
    std::string cmd = "python3 alert.py " + std::to_string(p) + " " + address + (test ? " --test" : "");
    if (system(cmd.c_str()) != 0) printf("alert.py failed - check cuda/alert.py and mail.env\n");
    printf("\a"); fflush(stdout);
}

// Known solved puzzle keys (puzzle, hex key) - checked against the real addresses.
static const struct { int p; const char* key; } KNOWN[] = {
    {1, "1"}, {2, "3"}, {3, "7"}, {4, "8"}, {5, "15"}, {6, "31"}, {7, "4c"}, {8, "e0"}, {9, "1d3"}, {10, "202"},
    {11, "483"}, {12, "a7b"}, {13, "1460"}, {14, "2930"}, {15, "68f3"}, {16, "c936"}, {17, "1764f"},
    {18, "3080d"}, {19, "5749f"}, {20, "d2c55"}, {21, "1ba534"}, {22, "2de40f"}, {23, "556e52"}, {24, "dc2a04"},
    {25, "1fa5ee5"}, {26, "340326e"}, {27, "6ac3875"}, {28, "d916ce8"}, {29, "17e2551e"}, {30, "3d94cd64"},
    {31, "7d4fe747"}, {32, "b862a62e"}, {33, "1a96ca8d8"}, {34, "34a65911d"}, {35, "4aed21170"},
    {36, "9de820a7c"}, {37, "1757756a93"}, {38, "22382facd0"}, {39, "4b5f8303e9"}, {40, "e9ae4933d6"},
    {41, "153869acc5b"}, {42, "2a221c58d8f"}, {43, "6bd3b27c591"}, {44, "e02b35a358f"}, {45, "122fca143c05"},
    {46, "2ec18388d544"}, {47, "6cd610b53cba"}, {48, "ade6d7ce3b9b"}, {49, "174176b015f4d"},
    {50, "22bd43c2e9354"}, {51, "75070a1a009d4"}, {52, "efae164cb9e3c"}, {53, "180788e47e326c"},
    {54, "236fb6d5ad1f43"}, {55, "6abe1f9b67e114"}, {56, "9d18b63ac4ffdf"}, {57, "1eb25c90795d61c"},
    {58, "2c675b852189a21"}, {59, "7496cbb87cab44f"}, {60, "fc07a1825367bbe"}, {61, "13c96a3742f64906"},
    {62, "363d541eb611abee"}, {63, "7cce5efdaccf6808"}, {64, "f7051f27b09112d4"}, {65, "1a838b13505b26867"},
    {66, "2832ed74f2b5e35ee"}, {67, "730fc235c1942c1ae"}, {68, "bebb3940cd0fc1491"}, {69, "101d83275fb2bc7e0c"},
    {70, "349b84b6431a6c4ef1"}, {75, "4c5ce114686a1336e07"}, {80, "ea1a5c66dcc11b5ad180"},
    {85, "11720c4f018d51b8cebba8"}, {90, "2ce00bb2136a445c71e85bf"}, {95, "527a792b183c7f64a0e8b1f4"},
    {100, "af55fc59c335c8ec67ed24826"}, {105, "16f14fc2054cd87ee6396b33df3"},
    {110, "35c0d7234df7deb0f20cf7062444"}, {115, "60f4d11574f5deee49961d9609ac6"},
    {120, "b10f22572c497a836ea187f2e1fc23"}, {125, "1c533b6bb7f0804e09960225e44877ac"},
    {130, "33e7665705359f04f28b88cf897c603c9"}, {135, "6d9392a16883f90903d5f78da57af07eb2"},
};

// Лестница: решаем решённые пазлы from..to по очереди «на глазах». Кандидат = старший бит 1
// (он у всех ключей пазла) + ровно (единиц ключа - 1) единиц в случайных позициях остальных n-1 бит.
// GPU: каждый поток тянет случайные n-1 бит и берёт только строки с нужным числом единиц
// (отбраковка = равномерный выбор из того же множества), сравнивает с ключом.
__device__ __forceinline__ u64 xs64(u64& s) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
__global__ void k_ladder(int n, int onesm1, u64 klo, u64 khi, u64 mlo, u64 mhi, u64 tlo, u64 thi, u32 iters, u64 seed,
                         int* found, u64* fc, int* best, u64* bc, unsigned long long* tries) {
    u64 s = seed + (u64)(blockIdx.x * blockDim.x + threadIdx.x) * 0x9E3779B97F4A7C15ull;   // splitmix64 → своё зерно потоку
    s = (s ^ (s >> 30)) * 0xBF58476D1CE4E5B9ull; s = (s ^ (s >> 27)) * 0x94D049BB133111EBull; s ^= s >> 31; if (!s) s = 1;
    int lb = -1; u64 blo = 0, bhi = 0; unsigned long long cnt = 0;
    for (u32 it = 0; it < iters; it++) {
        u64 rlo = xs64(s) & mlo, rhi = mhi ? (xs64(s) & mhi) : 0;
        if (__popcll(rlo) + __popcll(rhi) != onesm1) continue;
        cnt++;
        u64 clo = rlo | tlo, chi = rhi | thi;
        int m = n - __popcll(clo ^ klo) - __popcll(chi ^ khi);
        if (m > lb) { lb = m; blo = clo; bhi = chi; }
        if (m == n) { if (atomicExch(found, 1) == 0) { fc[0] = clo; fc[1] = chi; } break; }
        if ((it & 1023) == 0 && *(volatile int*)found) break;
    }
    atomicAdd(tries, cnt);
    if (lb > atomicMax(best, lb)) { bc[0] = blo; bc[1] = bhi; }   // только для показа: гонка допустима
}
static void ladder(int from, int to) {
    typedef unsigned __int128 u128;
    auto pc = [](u128 x) { return __builtin_popcountll((u64)x) + __builtin_popcountll((u64)(x >> 64)); };
    auto draw = [](int n, u128 c, u128 key, int best, u64 tries, double el) {
        std::string line;
        for (int i = n - 1; i >= 0; i--) {
            int b = (int)((c >> i) & 1);
            if (b == (int)((key >> i) & 1)) { line += "\x1b[92m"; line += (char)('0' + b); } else line += "\x1b[90m.";
        }
        printf("\r[%3d] %s\x1b[0m  %d/%d  попыток %.3e  %.1fM/с", n, line.c_str(), best, n, (double)tries, el > 0 ? tries / el / 1e6 : 0.0);
        fflush(stdout);
    };
    const int LB = 4096, LT = 256;   // 1M потоков
    int *dFound, *dBest; u64 *dFc, *dBc; unsigned long long* dTries;
    CK(cudaMalloc(&dFound, 4)); CK(cudaMalloc(&dBest, 4)); CK(cudaMalloc(&dFc, 16)); CK(cudaMalloc(&dBc, 16)); CK(cudaMalloc(&dTries, 8));
    auto all0 = std::chrono::steady_clock::now();
    for (int n = from; n <= to; n++) {
        const char* hex = nullptr; for (auto& e : KNOWN) if (e.p == n) hex = e.key;
        if (!hex) { printf("пазл %3d: не решён — ключа нет, сравнивать не с чем. Лестница остановлена.\n", n); break; }
        u32 kk[8]; key_hex(hex, kk);
        u128 key = 0; for (int w = 3; w >= 0; w--) key = (key << 32) | kk[w];
        int ones = pc(key), b = n - 1;   // b свободных бит под старшим
        u64 mlo = b >= 64 ? ~0ull : ((1ull << b) - 1), mhi = b > 64 ? ((1ull << (b - 64)) - 1) : 0;
        u64 tlo = b < 64 ? (1ull << b) : 0, thi = b >= 64 ? (1ull << (b - 64)) : 0;
        int zero = 0, minus = -1; unsigned long long z64 = 0;
        CK(cudaMemcpy(dFound, &zero, 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dBest, &minus, 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dTries, &z64, 8, cudaMemcpyHostToDevice));
        u32 iters = 64; int found = 0, best = -1; u128 bestc = 0; unsigned long long tries = 0;
        auto t0 = std::chrono::steady_clock::now();
        while (!found) {
            auto l0 = std::chrono::steady_clock::now();
            k_ladder<<<LB, LT>>>(n, ones - 1, (u64)key, (u64)(key >> 64), mlo, mhi, tlo, thi, iters, rng(), dFound, dFc, dBest, dBc, dTries);
            CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
            if (std::chrono::duration<double>(std::chrono::steady_clock::now() - l0).count() < 0.1 && iters < (1u << 24)) iters *= 2;   // ~0.1 с на запуск
            u64 c2[2]; CK(cudaMemcpy(&found, dFound, 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(&best, dBest, 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(c2, dBc, 16, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(&tries, dTries, 8, cudaMemcpyDeviceToHost));
            bestc = ((u128)c2[1] << 64) | c2[0];
            draw(n, bestc, key, best, tries, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        }
        u64 f2[2]; CK(cudaMemcpy(f2, dFc, 16, cudaMemcpyDeviceToHost));
        u128 fk = ((u128)f2[1] << 64) | f2[0];
        double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        draw(n, fk, key, n - pc(fk ^ key), tries, el);
        printf("  %s за %.2f с\n", fk == key ? "\x1b[92mрешён\x1b[0m" : "\x1b[91mОШИБКА: найденное != ключ\x1b[0m", el);
    }
    cudaFree(dFound); cudaFree(dBest); cudaFree(dFc); cudaFree(dBc); cudaFree(dTries);
    printf("лестница: всего %.1f с\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - all0).count());
}

static void usage(FILE* f) {
    fprintf(f,
"snake - CUDA brute-forcer for Bitcoin puzzle keys 71..160\n"
"\n"
"Usage:\n"
"  snake [--snake] [--watch] [--load PCT] [--seconds N] [--nocool] [--addr FILE]\n"
"  snake --spread P [--kmin K] [--kmax K] [--watch [--ref P] [--ones N]] [common]\n"
"  snake --exhaust P [--kmin K] [--kmax K] [--chunkfrom C] [--force] [common]\n"
"  snake --freeze P [--frz M] [--check] [--vis] [common]\n"
"  snake --test\n"
"  snake --ladder [--from A] [--to B]\n"
"  snake --show STEPS\n"
"\n"
"Modes:\n"
"  --snake          the snake walk (default): one 160-bit path visits every unsolved\n"
"                   puzzle 71..160 once per step; after a full pass the path shifts\n"
"                   one bit. This is the main search mode.\n"
"  --spread P       one puzzle P, random keys with a fixed popcount k (the kmin..kmax\n"
"                   slices are split across walkers).\n"
"  --exhaust P      deterministic, gap-free sweep of whole popcount slices of P.\n"
"  --freeze P       freeze M ones of P and sweep every remaining low bit; progress\n"
"                   is stored in freeze_masks_p<P>.txt.\n"
"  --test           built-in CPU/GPU self-tests (short and safe).\n"
"  --ladder         demo: solve known puzzles from..to in front of you.\n"
"  --show STEPS     print the first STEPS snake steps as keys, then exit.\n"
"  --watch          visualise walkers against a known puzzle key (with --snake or --spread).\n"
"\n"
"Common options:\n"
"  --load PCT       GPU duty cycle 5..100 (default 100)\n"
"  --seconds N      stop after N seconds (0 = run forever)\n"
"  --nocool         disable the 10-minute cooling break every 50 minutes\n"
"  --addr FILE      puzzle address list (default ../allPuzzles.js)\n"
"  --help, -h       this text\n"
"\n"
"Run from the cuda/ directory: the default address list is ../allPuzzles.js.\n");
}

int main(int argc, char** argv) {
    bool testOnly = false, alertTest = false, cool = true, watch = false, ladderOn = false, force = false, snakeOn = false; double seconds = 0;
    int show = 0, load = 100, ones = -1, ref = 70, lfrom = 1, lto = 71, spread = 0, kmin = 1, kmax = 70, exhaust = 0, freeze = 0, frz = 0;
    bool chk = false, vis = false;
    u64 chunkFrom = 0;
    std::string addrPath = "../allPuzzles.js";
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--help" || a == "-h") { usage(stdout); return 0; }
        else if (a == "--snake") snakeOn = true;   // explicit name for the default ladder walk
        else if (a == "--test") testOnly = true;
        else if (a == "--alert-test") alertTest = true;
        else if (a == "--nocool") cool = false;
        else if (a == "--watch") watch = true;   // визуализация против известного ключа 70, замедляет поиск
        else if (a == "--ones" && i + 1 < argc) ones = atoi(argv[++i]);   // watch-кандидат: нули + столько единиц в случайных позициях
        else if (a == "--ref" && i + 1 < argc) ref = atoi(argv[++i]);     // watch-эталон: номер решённого пазла (по умолчанию 70)
        else if (a == "--ladder") ladderOn = true;                        // лестница: решать пазлы --from..--to по очереди на глазах
        else if (a == "--spread" && i + 1 < argc) {
            spread = atoi(argv[++i]);
            if (spread < 3 || spread > 160) { fprintf(stderr, "--spread: puzzle 3..160\n"); return 1; }
        }
        else if (a == "--exhaust" && i + 1 < argc) {
            exhaust = atoi(argv[++i]);
            if (exhaust < 4 || exhaust > 128) { fprintf(stderr, "--exhaust: puzzle 4..128 (masks are 128-bit, the top bit is separate)\n"); return 1; }
        }
        else if (a == "--freeze" && i + 1 < argc) {
            freeze = atoi(argv[++i]);
            if (freeze < 3 || freeze > 128) { fprintf(stderr, "--freeze: puzzle 3..128 (masks are 128-bit, the top bit is separate)\n"); return 1; }
        }
        else if (a == "--frz" && i + 1 < argc) frz = atoi(argv[++i]);         // frozen ones for --freeze (default: half the low bits)
        else if (a == "--check") chk = true;                                  // --freeze on a solved puzzle: freeze the known key's own ones
        else if (a == "--vis") vis = true;                                    // --freeze: slow picture of the walk, no search, nothing written
        else if (a == "--chunkfrom" && i + 1 < argc) chunkFrom = strtoull(argv[++i], nullptr, 10);   // resume --exhaust of a single k
        else if (a == "--force") force = true;   // allow an --exhaust slice that needs more than a day of GPU time
        else if (a == "--kmin" && i + 1 < argc) kmin = atoi(argv[++i]);
        else if (a == "--kmax" && i + 1 < argc) kmax = atoi(argv[++i]);
        else if (a == "--from" && i + 1 < argc) lfrom = std::max(1, atoi(argv[++i]));
        else if (a == "--to" && i + 1 < argc) lto = std::min(127, atoi(argv[++i]));
        else if (a == "--load" && i + 1 < argc) load = std::max(5, std::min(100, atoi(argv[++i])));
        else if (a == "--show" && i + 1 < argc) show = atoi(argv[++i]);
        else if (a == "--seconds" && i + 1 < argc) seconds = atof(argv[++i]);
        else if (a == "--addr" && i + 1 < argc) addrPath = argv[++i];
        else { usage(stderr); return 1; }
    }
    if (snakeOn && (spread || exhaust || freeze || ladderOn)) {
        fprintf(stderr, "--snake is the default walk; do not combine it with --spread, --exhaust, --freeze or --ladder\n");
        return 1;
    }
    if (alertTest) { alarm(71, "1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU", true); return 0; }
    if (vis && !freeze) { fprintf(stderr, "--vis is a picture of --freeze: add --freeze P\n"); return 1; }
    u64 seed; if (getrandom(&seed, 8, 0) != 8) return 1; rng.seed(seed);

    // ---- CPU self-test: hash vectors, field, address decoding, known keys ----
    int fails = 0;
    auto check = [&](bool ok, const std::string& what) { printf("  [%s] %s\n", ok ? " OK " : "FAIL", what.c_str()); if (!ok) fails++; };
    printf("CPU self-test\n");
    { u8 d[32]; sha256_short((const u8*)"abc", 3, d); check(hexs(d, 32) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", "SHA-256(\"abc\")"); }
    { u8 d[20]; rmd160_short((const u8*)"", 0, d); check(hexs(d, 20) == "9c1185a5c5e9fc54612808977ee8f548b2258d31", "RIPEMD-160(\"\")"); }
    { u8 d[20]; rmd160_short((const u8*)"abc", 3, d); check(hexs(d, 20) == "8eb208f7e05d987a9b044a8e98c6b087f15a0bfc", "RIPEMD-160(\"abc\")"); }
    { fe a = fe_hex("deadbeefcafebabe0123456789abcdef00112233445566778899aabbccddeeff"), b, c; fe_inv(b, a); fe_mul(c, a, b);
      check(fe_eq(c, fe{{1, 0, 0, 0, 0, 0, 0, 0}}), "a * a^-1 == 1 (mod p)"); }
    { u32 k[8]; key_hex("1", k); check(wif(k) == "KwDiBf89QgGbjEhKnhXJuH7LrciVrZi3qYjgd9M7rFU73sVHnoWn", "WIF of key 1 (compressed)"); }

    std::vector<std::string> addr(161);
    { std::ifstream f(addrPath); std::string line; int n = 0;
      while (n < 160 && std::getline(f, line)) { while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back(); addr[++n] = line; }
      if (n != 160) { fprintf(stderr, "cannot read 160 addresses from %s\n", addrPath.c_str()); return 1; } }
    static u32 tgt[161][5] = {};
    { bool ok = true; for (int p = 1; p <= 160; p++) ok &= b58_h160(addr[p], tgt[p]); check(ok, "160 puzzle addresses decode, Base58Check checksums valid"); }

    hpow2[0] = {fe_hex("79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798"),
                fe_hex("483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8")};
    for (int k = 1; k <= 160; k++) { jac P = to_jac(hpow2[k - 1]); jdbl(P); hpow2[k] = to_aff(P); }
    std::vector<int> verified;  // puzzles whose key -> address matched
    for (auto& kn : KNOWN) {
        u32 k[8], h[5]; key_hex(kn.key, k); key_h160(k, h);
        bool ok = !memcmp(h, tgt[kn.p], 20);
        check(ok, "puzzle " + std::to_string(kn.p) + " key " + kn.key + " -> " + addr[kn.p]);
        if (ok && kn.p >= 71) verified.push_back(kn.p);
    }
    if (fails) { printf("CPU self-test FAILED (%d)\n", fails); return 1; }

    // ---- --freeze P --vis: a slow, explained picture of the walk. Pure CPU: no CUDA context, no search,
    // ---- no hashing, nothing written - so it is safe to watch while a real search is running. ----
    if (freeze && vis) {
        int p = freeze;
        if (p < 3 || p > 128) { fprintf(stderr, "--vis: puzzle 3..128\n"); return 1; }
        int nb = p - 1;
        int m = frz > 0 ? frz : nb / 2 + 1;
        if (m < 1 || m > nb) { fprintf(stderr, "--vis: --frz 1..%d for puzzle %d\n", nb, p); return 1; }
        for (auto& kn : KNOWN) if (kn.p == p) {                    // solved puzzle: same cap as the search
            u32 k2[8]; key_hex(kn.key, k2); int q = 0;
            for (int b = 0; b < nb; b++) q += getbit(k2, b);
            m = std::min(m, q);
        }
        int f = nb - m;
        u128 total = (u128)1 << f;
        auto rand_mask = [&]() {
            u128 v = 0;
            for (int c = 0; c < m; ) { int b = (int)(rng() % nb); u128 bit = bit128h(b); if (!(v & bit)) { v |= bit; c++; } }
            return v;
        };
        auto free_of = [&](u128 frozen) { u128 v = 0; for (int b = 0; b < nb; b++) if (!(frozen & bit128h(b))) v |= bit128h(b); return v; };
        FILE* dtty = fopen("/dev/tty", "w");
        FILE* out = dtty ? dtty : stdout;
        static const char* why[] = {
            "a puzzle-P key is 2^(P-1) + the unknown low bits, so the top bit is always 1",
            "this mode freezes M of the low bits as ones (yellow): they never move while the mask runs",
            "the other f = (P-1)-M places are free: the sweep walks all 2^f patterns of them, each exactly once",
            "one step = one rank, like counting in binary over the free places: one or two bits flip",
            "each flipped bit is one point addition on the GPU; the running point stays key*G, no scalar multiply",
            "k is the ones count of the key; this mode can only produce k = M+1 .. P",
            "a mask is one choice of the frozen places: C(P-1,M) of them exist, and they overlap heavily",
            "when every mask is done the search reports: all combinations were exhausted",
        };
        const int nWhy = (int)(sizeof why / sizeof why[0]);
        u64 shown = total > 512 ? 512 : (u64)total;         // steps shown per mask (a real mask is 2^f keys)
        u64 nMask = 1, r = 0, frame = 0;
        u128 B = rand_mask(), F = free_of(B), s = 0, prev = 0;
        std::string note;
        auto tVis = std::chrono::steady_clock::now();
        printf("Freeze visualisation: a picture only - nothing is searched, hashed or written. Ctrl-C stops.\n"
               "(-freeze %d, %d frozen ones, %d free places, 2^%d = %s keys per mask; --seconds N to time-limit)\n",
               p, m, f, f, u128_str(total).c_str());
        for (;;) {
            u128 M = B | s;
            std::string key = "\x1b[97m1\x1b[0m";                        // the top bit is always 1
            int lastc = -1;
            for (int b = nb - 1; b >= 0; b--) {
                int c = (B & bit128h(b)) ? 3 : ((M & bit128h(b)) ? 2 : 1);   // 3 frozen, 2 free=1, 1 free=0
                if (c != lastc) { key += c == 3 ? "\x1b[93m" : c == 2 ? "\x1b[92m" : "\x1b[90m"; lastc = c; }
                key += c == 1 ? '0' : '1';
            }
            int k = 1 + (int)__builtin_popcountll((u64)M) + (int)__builtin_popcountll((u64)(M >> 64));
            std::string flips; int nflip = 0;
            for (int b = 0; b < nb; b++) if ((M ^ prev) & bit128h(b)) {
                if (!flips.empty()) flips += ", ";
                flips += "bit " + std::to_string(b) + ((M & bit128h(b)) ? " on" : " off");
                nflip++;
            }
            std::string fr;
            fr += "FREEZE visualisation  -  picture only, nothing is searched or written  (Ctrl-C stops)\n\n";
            fr += "puzzle " + std::to_string(p) + ":  key = 2^" + std::to_string(nb) + " + low bits.   low bits " + std::to_string(nb) +
                  ",  frozen ones " + std::to_string(m) + ",  free places " + std::to_string(f) + "\n";
            fr += "yellow = frozen 1 (never moves)    green = free place holding 1    grey = free place holding 0\n";
            fr += "\n  " + key + "\x1b[0m\n\n";
            fr += "mask " + std::to_string(nMask) + ":  step " + u128_str(r) + " of " + u128_str(shown);
            fr += total > shown ? " shown   (one real mask = 2^" + std::to_string(f) + " = " + u128_str(total) + " keys)\n"
                                : "   (the whole mask)\n";
            fr += flips.empty() ? std::string("first key of this mask: the frozen ones are fixed, all free places are 0\n")
                                : "changed: " + flips + "  ->  " + std::to_string(nflip) + (nflip == 1 ? " point add" : " point adds") + " on the GPU\n";
            fr += "ones: k = " + std::to_string(k) + "   (this mode can only make k = " + std::to_string(m + 1) + " .. " + std::to_string(nb + 1) + ")\n\n";
            if (!note.empty()) fr += note + "\n";
            fr += std::string("\x1b[96m* ") + why[(frame / 12) % nWhy] + "\x1b[0m\n";
            if (dtty) fprintf(out, "\x1b[H\x1b[J%s", fr.c_str()); else fprintf(out, "%s", fr.c_str());
            fflush(out);
            prev = M;
            if (++r >= shown) {                            // this mask is done (or its shown part): next random mask
                note = total <= shown ? "mask complete: every one of its " + u128_str(total) + " keys was shown exactly once"
                                      : "shown the first " + u128_str((u128)shown) + " steps; a real mask continues to " + u128_str(total);
                r = 0; s = 0; prev = 0; nMask++;
                B = rand_mask(); F = free_of(B);
            } else s = (s - F) & F;
            frame++;
            if (seconds > 0 && std::chrono::duration<double>(std::chrono::steady_clock::now() - tVis).count() >= seconds) return 0;
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
        }
    }

    // ---- tables -> GPU ----
    std::vector<aff> win(9 * 256);
    for (int w = 0; w < 9; w++) {
        win[w * 256 + 1] = hpow2[8 * w];
        for (int j = 2; j < 256; j++) win[w * 256 + j] = aff_add(win[w * 256 + j - 1], hpow2[8 * w]);
    }
    aff* dwin; CK(cudaMalloc(&dwin, win.size() * sizeof(aff)));
    CK(cudaMemcpy(dwin, win.data(), win.size() * sizeof(aff), cudaMemcpyHostToDevice));
    CK(cudaMemcpyToSymbol(g_win, &dwin, sizeof(dwin)));
    CK(cudaMemcpyToSymbol(c_pow2, hpow2, sizeof(hpow2)));
    CK(cudaMemcpyToSymbol(c_tgt, tgt, sizeof(tgt)));

    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    const int TPB = 256, blocks = prop.multiProcessorCount * 8;
    const size_t NT = (size_t)blocks * TPB;
    u32 *dV, *dHitN; u64* dR; Hit* dHits;
    CK(cudaMalloc(&dV, NT * 20)); CK(cudaMalloc(&dR, NT * 8)); CK(cudaMalloc(&dHitN, 4)); CK(cudaMalloc(&dHits, 1024 * sizeof(Hit)));
    std::vector<u32> hV(NT * 5); std::vector<u64> hR(NT);
    auto reset_threads = [&]() {
        for (size_t t = 0; t < NT; t++) { random_V(&hV[t * 5]); hR[t] = rng() | 1; }
    };
    auto upload = [&]() {
        CK(cudaMemcpy(dV, hV.data(), NT * 20, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dR, hR.data(), NT * 8, cudaMemcpyHostToDevice));
        CK(cudaMemset(dHitN, 0, 4));
    };
    auto read_hits = [&](std::vector<Hit>& out) {
        u32 n; CK(cudaMemcpy(&n, dHitN, 4, cudaMemcpyDeviceToHost)); n = n > 1024 ? 1024 : n;
        out.resize(n); if (n) CK(cudaMemcpy(out.data(), dHits, n * sizeof(Hit), cudaMemcpyDeviceToHost));
    };
    printf("GPU: %s, %d SMs, %zu threads\n", prop.name, prop.multiProcessorCount, NT);

    // ---- GPU field multiply (PTX) vs CPU field multiply (plain C) ----
    {
        const int N = 1 << 20;
        std::vector<fe> A(N), B(N), R(N);
        const char* edge[] = {"0", "1", "2", "3d1", "1000003d1",
            "fffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2e",   // p-1
            "fffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2d",   // p-2
            "fffffffffffffffffffffffffffffffffffffffffffffffffffffffefffff000",
            "ffffffffffffffffffffffffffffffff00000000000000000000000000000000",
            "8000000000000000000000000000000000000000000000000000000000000000",
            "00000000ffffffffffffffffffffffffffffffffffffffffffffffffffffffff"};
        const int NE = sizeof(edge) / sizeof(edge[0]);
        for (int i = 0; i < N; i++) {
            if (i < NE * NE) { A[i] = fe_hex(edge[i / NE]); B[i] = fe_hex(edge[i % NE]); continue; }
            for (int j = 0; j < 8; j++) { A[i].v[j] = (u32)rng(); B[i].v[j] = (u32)rng(); }
            if (i & 1) for (int j = 2; j < 8; j++) A[i].v[j] = B[i].v[j] = 0xFFFFFFFFu;   // stress the carries
            while (fe_gep(A[i])) A[i].v[7] >>= 1;
            while (fe_gep(B[i])) B[i].v[7] >>= 1;
        }
        fe *dA, *dB, *dO;
        CK(cudaMalloc(&dA, N * 32)); CK(cudaMalloc(&dB, N * 32)); CK(cudaMalloc(&dO, N * 32));
        CK(cudaMemcpy(dA, A.data(), N * 32, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dB, B.data(), N * 32, cudaMemcpyHostToDevice));
        k_mul_test<<<N / 256, 256>>>(dA, dB, dO, N); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(R.data(), dO, N * 32, cudaMemcpyDeviceToHost));
        cudaFree(dA); cudaFree(dB); cudaFree(dO);
        int bad = 0;
        for (int i = 0; i < N; i++) { fe c; fe_mul(c, A[i], B[i]); if (!fe_eq(c, R[i])) { if (bad < 3) printf("  mismatch %s * %s\n", fe_str(A[i]).c_str(), fe_str(B[i]).c_str()); bad++; } }
        printf("GPU field multiply vs CPU: %d / %d mismatches %s\n", bad, N, bad ? "FAILED" : "PASSED");
        if (bad) return 1;
    }

    // ---- GPU self-test: plant real solved keys, GPU must find them from the real addresses ----
    {
        printf("GPU self-test (targets = all 90 real addresses 71..160)\n");
        u8 act[161] = {}; for (int p = 71; p <= 160; p++) act[p] = 1;
        CK(cudaMemcpyToSymbol(c_active, act, sizeof(act)));
        reset_threads();
        const int STEPS = 8;
        std::set<std::tuple<u32, u32, u32>> expect;   // (tid, puzzle, step)
        for (size_t i = 0; i < verified.size(); i++) {
            int p = verified[i];
            u32 tid = (u32)((1237 + 7919 * i * 13) % NT);
            int s = std::min<int>(i % STEPS, 160 - p);   // where in the snake it shows up
            u32 k[8]; for (auto& kn : KNOWN) if (kn.p == p) key_hex(kn.key, k);
            u32* V = &hV[tid * 5];
            for (int b = 0; b < 159; b++) setbit(V, b, rng() & 1);
            for (int b = 0; b < p - 1; b++) setbit(V, b + s, getbit(k, b));   // after s shifts V mod 2^(p-1) == key - 2^(p-1)
            expect.insert({tid, (u32)p, (u32)s});
            printf("  planted puzzle %d key at thread %u, snake step %d\n", p, tid, s);
        }
        upload();
        auto t0 = std::chrono::steady_clock::now();
        k_snake<<<blocks, TPB>>>(dV, dR, STEPS, dHitN, dHits); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::vector<Hit> hits; read_hits(hits);
        std::set<std::tuple<u32, u32, u32>> got; int badKey = 0;
        for (auto& h : hits) {
            got.insert({h.tid, h.p, h.step});
            u32 k[8], hh[5]; V_to_key(h.V, h.p, k); key_h160(k, hh);
            bool ok = !memcmp(hh, tgt[h.p], 20);
            if (!ok) badKey++;
            printf("  GPU hit: puzzle %u thread %u step %u key %s %s\n", h.p, h.tid, h.step, key_str(k).c_str(), ok ? "(CPU re-check OK)" : "(CPU re-check FAILED)");
        }
        if (!hits.empty()) {
            u32 k[8], hh[5]; V_to_key(hits[0].V, hits[0].p, k); key_h160(k, hh);
            printf("  sample found.txt record (from a GPU test hit, not written to disk):\n%s", found_record(hits[0].p, k, addr[hits[0].p], !memcmp(hh, tgt[hits[0].p], 20)).c_str());
        }
        bool ok = got == expect && badKey == 0 && hits.size() == expect.size();
        printf("  %zu planted, %zu found, %.2fs for %.1fM keys\n", expect.size(), hits.size(), dt, NT * STEPS * 90 / 1e6);
        printf("GPU self-test %s\n", ok ? "PASSED" : "FAILED");
        if (!ok) return 1;
    }
    if (testOnly) return 0;
    if (ladderOn) { ladder(lfrom, lto); return 0; }

    // ---- --exhaust P: whole fixed-popcount slices, every key exactly once ----
    if (exhaust) {
        int p = exhaust;
        if (p < 4 || p > 128) { fprintf(stderr, "--exhaust: puzzle 4..128 (masks are 128-bit, the top bit is separate)\n"); return 1; }
        int nb = p - 1;
        {   // exact binomials for the GPU unranking (saturated: only compared with ranks < 2^62)
            static u64 cb[129][36];
            for (int n = 0; n <= 128; n++) for (int k = 0; k <= 35; k++) {
                u128 v = binom128(n, k); cb[n][k] = v > (u128)(u64)~0ull ? ~0ull : (u64)v;
            }
            CK(cudaMemcpyToSymbol(c_bin, cb, sizeof cb));
        }
        const u64 CS = 8192;                       // keys per chunk (one chunk per thread per launch)
        auto chunks_of = [&](u64 total) { return (total + CS - 1) / CS; };
        auto base_jac = [&](bool inv) -> jac {     // start point: 2^(p-1)*G, or (2^p - 1)*G when enumerating zeros
            if (!inv) return to_jac(hpow2[p - 1]);
            aff t = aff_add(hpow2[p - 1], hpow2[p - 1]);
            aff mg = hpow2[0]; fe zz = {}; fe_sub(mg.y, zz, mg.y);
            return to_jac(aff_add(t, mg));
        };
        auto launch = [&](int z, bool inv, u64 total, u64 first, u64 count, std::vector<Hit>& out) {
            CK(cudaMemset(dHitN, 0, 4));
            k_exhaust<<<blocks, TPB>>>(p, nb, z, first, count, CS, total, base_jac(inv), inv ? 1 : 0, dHitN, dHits);
            CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
            read_hits(out);
        };
        auto mask_of = [](const Hit& h) {
            return ((u128)h.V[3] << 96) | ((u128)h.V[2] << 64) | ((u128)h.V[1] << 32) | (u128)h.V[0];
        };
        auto key_of = [&](u128 mask, bool inv) {   // enumerated mask -> full p-bit key
            u128 key = 0;
            for (int b = 0; b < nb; b++) key |= bit128h(b);
            if (inv) key &= ~mask; else key = mask;
            return key | bit128h(p - 1);
        };
        auto key_words = [](u128 key, u32 kk[8]) {   // u128 -> the 8 u32 words key_h160 wants
            u64 lo = (u64)key, hi = (u64)(key >> 64);
            kk[0] = (u32)lo; kk[1] = (u32)(lo >> 32); kk[2] = (u32)hi; kk[3] = (u32)(hi >> 32);
            for (int i = 4; i < 8; i++) kk[i] = 0;
        };

        printf("%s", BANNER);
        printf("Exhaust: puzzle %d, whole slices ones %d..%d (every key once, no random walk)\n", p, kmin, kmax);

        // deterministic self-test: plant a 3-ones key and a 3-zeros key, both must be found
        {
            u32 saved[5]; memcpy(saved, tgt[p], 20);
            const int tb_small[2][3] = { {0, 1, 2}, {0, 1, 2} };
            int tb[2][3];
            for (int t = 0; t < 2; t++) for (int j = 0; j < 3; j++) tb[t][j] = tb_small[t][j];
            if (nb >= 7) {   // a spread of positions, as before; keep all three distinct
                tb[0][0] = 1; tb[0][1] = nb / 3; tb[0][2] = nb - 1;
                tb[1][0] = 2; tb[1][1] = nb / 2 + 1; tb[1][2] = nb - 2;
            }
            bool ok = true;
            for (int t = 0; t < 2; t++) {
                bool inv = t == 1;
                u128 M = bit128h(tb[t][0]) | bit128h(tb[t][1]) | bit128h(tb[t][2]);
                u128 key = key_of(M, inv);
                u32 kk[8]; key_words(key, kk);
                key_h160(kk, tgt[p]);
                CK(cudaMemcpyToSymbol(c_tgt, tgt, sizeof(tgt)));
                u64 total = (u64)binom128(nb, 3);
                std::vector<Hit> hits; launch(3, inv, total, 0, chunks_of(total), hits);
                bool found = false;
                for (auto& h : hits) found |= mask_of(h) == M;
                printf("  exhaust self-test (%s, %s keys): %s  [want mask %016llx%016llx, %zu hits",
                       inv ? "3 zeros" : "3 ones", u128_str(total).c_str(), found ? "OK" : "FAILED",
                       (unsigned long long)(u64)(M >> 64), (unsigned long long)(u64)M, hits.size());
                for (auto& h : hits) printf(" %016llx%016llx", (unsigned long long)(u64)(mask_of(h) >> 64), (unsigned long long)(u64)mask_of(h));
                printf("]\n");
                ok &= found;
            }
            memcpy(tgt[p], saved, 20); CK(cudaMemcpyToSymbol(c_tgt, tgt, sizeof(tgt)));
            if (!ok) { fprintf(stderr, "exhaust self-test FAILED - refusing to run\n"); return 1; }
        }

        FILE* rep = fopen("exhaust_results.txt", "a");
        auto say = [&](const std::string& s) {
            printf("%s", s.c_str()); fflush(stdout);
            if (rep) { fputs(s.c_str(), rep); fflush(rep); }
        };
        { char when[64]; time_t t = time(nullptr); strftime(when, sizeof when, "%Y-%m-%d %H:%M:%S", localtime(&t));
          say(std::string("\n=== ") + when + "  exhaust puzzle " + std::to_string(p) + "  ones " + std::to_string(kmin) + ".." + std::to_string(kmax) + " ===\n"); }
        u64 grand = 0;
        bool anyHit = false;
        bool known = false; for (auto& e : KNOWN) known |= e.p == p;   // solved puzzle: a hit is a test, no alarm
        for (int k = kmin; k <= kmax; k++) {
            int w = k - 1;
            if (w < 0 || w > nb) { printf("slice k=%d: impossible for puzzle %d (free bits %d)\n", k, p, nb); continue; }
            int z = std::min(w, nb - w);
            bool inv = w > nb - w;
            u128 Nt = binom128(nb, z);
            if (z > 35 || Nt > ((u128)1 << 62)) {
                char b[256]; snprintf(b, sizeof b, "slice k=%d: C(%d,%d) = %s keys - too big for an exhaustive run, skipped\n", k, nb, z, u128_str(Nt).c_str());
                say(b); continue;
            }
            {   // a slice that needs more than a couple of hours of GPU time is almost certainly a typo, not a plan
                double days = (double)(u64)Nt / 2.0e8 / 86400.0;
                if (!force && days > 2.0 / 24.0) {
                    char b[256]; snprintf(b, sizeof b, "slice k=%d: %s keys - an exhaustive run would need ~%.2f GPU-days; skipped (add --force to run anyway)\n", k, u128_str(Nt).c_str(), days);
                    say(b); continue;
                }
            }
            u64 total = (u64)Nt, chunks = chunks_of(total);
            u64 startChunk = (kmin == kmax ? chunkFrom : 0);
            if (startChunk > chunks) startChunk = chunks;
            u64 first = startChunk;
            u64 skipped = std::min<u64>(total, startChunk * CS);
            u64 remaining = total - skipped;
            char b[512];
            if (remaining == 0) {
                snprintf(b, sizeof b, "slice k=%2d: nothing to do (start chunk %llu is past the end; %s keys skipped)\n",
                         k, (unsigned long long)startChunk, u128_str((u128)skipped).c_str());
                status_end();
                say(b);
                continue;
            }
            snprintf(b, sizeof b, "slice k=%2d: %s keys (%d free %s among %d low bits%s)\n", k, u128_str(Nt).c_str(),
                                  z, inv ? "zeros" : "ones", nb, inv ? ", key = (2^p - 1) - mask" : "");
            if (startChunk) {
                snprintf(b + strlen(b), sizeof b - strlen(b),
                         "  resumed at chunk %llu: %s keys already skipped, %s left\n",
                         (unsigned long long)startChunk, u128_str((u128)skipped).c_str(), u128_str((u128)remaining).c_str());
            }
            status_end();
            say(b);
            auto t0 = std::chrono::steady_clock::now(), lastRep = t0;
            u64 checked = 0;
            std::vector<Hit> hits;
            while (first < chunks) {
                u64 cnt = std::min<u64>(chunks - first, (u64)NT);
                launch(z, inv, total, first, cnt, hits);
                first += cnt;
                checked = std::min<u64>(remaining, (first - startChunk) * CS);
                for (auto& h : hits) {
                    u128 key = key_of(mask_of(h), inv);
                    u32 kk[8]; key_words(key, kk);
                    u32 hh[5]; key_h160(kk, hh);
                    bool ok = !memcmp(hh, tgt[p], 20);
                    std::string rec = found_record(p, kk, addr[p], ok);
                    if (!known) { FILE* f = fopen("found.txt", "a"); if (f) { fputs(rec.c_str(), f); fclose(f); } }
                    status_end();
                    printf("\n*** CANDIDATE in slice k=%d (exhaustive) ***\n%s", k, rec.c_str()); fflush(stdout);
                    say(rec); anyHit = true;
                    if (ok && known) {   // solved puzzle: the real address was found, so the walk is proven
                        printf("\n%s test hit in slice k=%d - exhaustive walk verified on a solved puzzle\n", "\x1b[92mOK\x1b[0m", k);
                        if (rep) fclose(rep);
                        return 0;
                    }
                    if (ok) { alarm(p, addr[p], false); if (rep) fclose(rep); return 0; }
                }
                auto now = std::chrono::steady_clock::now();
                double el = std::chrono::duration<double>(now - t0).count();
                if (std::chrono::duration<double>(now - lastRep).count() >= 10) {
                    lastRep = now;
                    snprintf(b, sizeof b, "slice k=%2d: %s / %s keys  %.2f%s  %.1f Mkeys/s", k, u128_str((u128)checked).c_str(),
                             u128_str((u128)remaining).c_str(), 100.0 * (double)checked / (double)remaining, "%", (double)checked / el / 1e6);
                    status_printf("%s", b);                                       // one line, rewritten in place
                    if (rep) { fputs(b, rep); fputc('\n', rep); fflush(rep); }    // the log still gets one line per update
                    if (load < 100) std::this_thread::sleep_for(std::chrono::duration<double>(1.0 * (100 - load) / load));
                }
            }
            double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            grand += checked;
            snprintf(b, sizeof b, "slice k=%2d: %s / %s keys checked in %.1f s - %s%s\n", k, u128_str((u128)checked).c_str(), u128_str((u128)remaining).c_str(), el,
                     checked == remaining ? "\x1b[92mno key found\x1b[0m" : "\x1b[93mINCOMPLETE\x1b[0m",
                     startChunk ? " (resumed: earlier chunks skipped)" : "");
            status_end();
            say(b);
        }
        char b[512]; snprintf(b, sizeof b, "TOTAL: %s keys checked, %s\n", u128_str((u128)grand).c_str(), anyHit ? "candidates above" : "no key found in any slice");
        status_end();
        say(b);
        if (rep) fclose(rep);
        return 0;
    }

    // ---- --freeze P: m ones frozen at chosen places, every other low bit swept (mask list in freeze_masks.txt) ----
    if (freeze) {
        int p = freeze;
        if (p < 3 || p > 128) { fprintf(stderr, "--freeze: puzzle 3..128 (masks are 128-bit, the top bit is separate)\n"); return 1; }
        int nb = p - 1;
        if (frz <= 0) frz = nb / 2 + 1;                    // default: half the low bits (puzzle 71 -> 36, puzzle 20 -> 10)
        if (frz < 1 || frz > nb) { fprintf(stderr, "--frz: 1..%d for puzzle %d\n", nb, p); return 1; }
        const u64 CS = 8192;                               // keys per chunk (one chunk per thread per launch)

        // a solved puzzle is searched exactly like a big one (random masks); --check freezes the ones of
        // the real key instead, so the sweep is guaranteed to contain it and the run must find it
        u32 kk[8] = {}; int q = 0; bool known = false;
        for (auto& kn : KNOWN) if (kn.p == p) { known = true; key_hex(kn.key, kk); }
        for (int b = 0; b < nb; b++) q += getbit(kk, b);
        int m = known ? std::min(frz, q) : frz;            // solved puzzle: never more frozen ones than the key has
        if (m < 1) { fprintf(stderr, "--freeze: the known key of puzzle %d has no ones below the top bit\n", p); return 1; }
        int f = nb - m;                                    // free places
        if (f > 62) { fprintf(stderr, "--freeze: 2^%d keys per mask does not fit a run; use --frz %d or more\n", f, nb - 62); return 1; }
        u64 total = (u64)1 << f, chunks = (total + CS - 1) / CS;
        double est = (double)total / 2.0e8;                // seconds per mask at the measured ~200 Mkeys/s
        u128 combos = nb <= 70 ? binom128(nb, m) : 0;      // masks in this (puzzle, frozen ones) space; 0 = uncountable
        u128 winMasks = (known && combos) ? binom128(q, m) : 0;   // masks of that space that contain the known key's ones
        auto hitChance = [&](u64 doneMasks) {              // how likely the key is in a mask already swept
            if (!winMasks || !combos) return -1.0;
            return 100.0 * (double)doneMasks * (double)winMasks / (double)combos;
        };
        auto rand_mask = [&]() {                           // m distinct places among nb
            u128 v = 0;
            for (int c = 0; c < m; ) { int b = (int)(rng() % nb); u128 bit = bit128h(b); if (!(v & bit)) { v |= bit; c++; } }
            return v;
        };
        auto free_of = [&](u128 frozen) { u128 v = 0; for (int b = 0; b < nb; b++) if (!(frozen & bit128h(b))) v |= bit128h(b); return v; };

        if (!force && est > 7200.0) {
            fprintf(stderr, "--freeze: %d ones frozen of %d -> %s keys per mask (~%.1f GPU-hours); use --frz %d or more, or --force\n",
                    m, nb, u128_str(total).c_str(), est / 3600.0, nb - 40);
            return 1;
        }
        if (!known && !force && f < 20) {                  // a mask list this small would spin the database
            fprintf(stderr, "--freeze: 2^%d keys per mask is too small to search with; use --frz %d or less\n", f, nb - 20);
            return 1;
        }
        int skipDone = known ? 0 : 1;                      // already-exhausted k slices are not hashed again

        auto key_of = [&](u128 mask) { return mask | bit128h(p - 1); };   // mask = the low p-1 bits
        auto mask_of = [](const Hit& h) { return ((u128)h.V[3] << 96) | ((u128)h.V[2] << 64) | ((u128)h.V[1] << 32) | (u128)h.V[0]; };
        auto key_words = [](u128 key, u32 w[8]) {
            u64 lo = (u64)key, hi = (u64)(key >> 64);
            w[0] = (u32)lo; w[1] = (u32)(lo >> 32); w[2] = (u32)hi; w[3] = (u32)(hi >> 32);
            for (int i = 4; i < 8; i++) w[i] = 0;
        };
        jac base = to_jac(hpow2[p - 1]);                   // bit p-1 * G; the mask bits are added on the GPU
        auto launch = [&](u128 B, u128 F, u64 first, u64 cnt, u64 tot, int skip, std::vector<Hit>& out) {
            CK(cudaMemset(dHitN, 0, 4));
            k_freeze<<<blocks, TPB>>>(p, (u64)B, (u64)(B >> 64), (u64)F, (u64)(F >> 64), first, cnt, CS, tot, base, skip, dHitN, dHits);
            CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
            read_hits(out);
        };
        auto hexmask = [](u128 v) {                          // fixed 32 hex digits: the database format
            char hx[33]; snprintf(hx, sizeof hx, "%016llx%016llx", (unsigned long long)(u64)(v >> 64), (unsigned long long)(u64)v);
            return std::string(hx);
        };
        auto bstr = [](u128 v, int bits) {                   // bits, high bit first (1 = set)
            std::string s;
            for (int b = bits - 1; b >= 0; b--) s += (v & bit128h(b)) ? '1' : '0';
            return s;
        };
        auto keyhex = [&](u128 mask) {                        // the key in hex at its own width, no zero padding
            char hx[33]; u128 key = mask | bit128h(p - 1);
            snprintf(hx, sizeof hx, "%016llx%016llx", (unsigned long long)(u64)(key >> 64), (unsigned long long)(u64)key);
            std::string t(hx); size_t z = t.find_first_not_of('0');
            return z == std::string::npos ? std::string("0") : t.substr(z);
        };

        printf("%s", BANNER);
        printf("Freeze: puzzle %d, %d of the %d low bits frozen as ones, 2^%d = %s keys swept per mask (~%.1f min)\n",
               p, m, nb, f, u128_str(total).c_str(), est / 60.0);
        if (known && chk) printf("  solved puzzle, --check: the frozen ones are taken from the known key, so this run must find it\n");
        else if (known) printf("  solved puzzle: masks are random like on 71, the known key gets no advantage (use --check to verify it instead)\n");
        if (known && winMasks)
            printf("  the key is reachable: %s of the %s masks contain its %d ones (%.5f%%, about 1 mask in %.0f)\n",
                   u128_str(winMasks).c_str(), u128_str(combos).c_str(), q, 100.0 * (double)winMasks / (double)combos,
                   (double)combos / (double)winMasks);
        else if (!known && m >= 12)
            printf("  the key is only reachable if it has at least %d ones; keys with fewer are invisible to this mode\n", m + 1);

        // per-ones-count table, like --spread: one row per k this mode can produce (drawn once, it never changes)
        {
            printf("ones   keys in this k              what happens here\n");
            for (int k = m + 1; k <= nb + 1; k++) {
                bool done = skipDone && (k <= 11 || k >= 61);
                printf("%4d   %-22s   %s\n", k, nb <= 70 ? u128_str(binom128(nb, k - 1)).c_str() : "-",
                       done ? "already exhausted, skipped" : ((known && chk) ? "swept (a hit is expected)" : "swept by the frozen masks"));
            }
            if (m >= 12 && m < nb) printf("       k = %d..%d are never covered by --frz %d (a key needs at least %d ones here)\n", 12, m, m, m + 1);
        }

        // self-test 1 (CPU): the rank walk must visit every subset of the free places, in ascending order, and
        // scatter_h must be its inverse.  self-test 2 (GPU): one chunk must find a key planted at a known rank.
        // self-test 3 (GPU): a key of an already-exhausted k must not be hashed when the skip is on.
        {
            bool ok = true;
            int fb = std::min(16, nb);
            u128 Fs = 0;
            for (int c = 0; c < fb; ) { int b = (int)(rng() % nb); u128 bit = bit128h(b); if (!(Fs & bit)) { Fs |= bit; c++; } }
            u128 s = 0; u64 c = 0;
            for (;;) {
                if (s != scatter_h(c, Fs, nb)) ok = false;
                u128 nx = (s - Fs) & Fs;
                if (nx && nx <= s) ok = false;
                if (!nx) break;
                s = nx; c++;
            }
            ok &= c + 1 == ((u64)1 << fb);
            printf("  freeze self-test (rank walk, %d free places): %s\n", fb, ok ? "OK" : "FAILED");
            u32 saved[5]; memcpy(saved, tgt[p], 20);
            for (int t = 0; t < 2; t++) {
                u64 r = t ? std::min<u64>(5000, total - 1) : 0;
                u128 Bt = rand_mask(), Ft = free_of(Bt);
                u128 planted = Bt | scatter_h(r, Ft, nb);
                u32 w[8]; key_words(key_of(planted), w); key_h160(w, tgt[p]);
                CK(cudaMemcpyToSymbol(c_tgt, tgt, sizeof(tgt)));
                std::vector<Hit> hits; launch(Bt, Ft, 0, 1, total, 0, hits);
                bool found = false;
                for (auto& h : hits) found |= mask_of(h) == planted;
                printf("  freeze self-test (GPU chunk, planted rank %llu): %s  [%zu hits, mask %s]\n",
                       (unsigned long long)r, found ? "OK" : "FAILED", hits.size(), hexmask(planted).c_str());
                ok &= found;
            }
            if (skipDone && nb + 1 >= 61 && total > CS) {      // all-ones low mask: k = nb+1, already exhausted
                u128 Bt = rand_mask(), Ft = free_of(Bt), planted = Bt | Ft;
                u32 w[8]; key_words(key_of(planted), w); key_h160(w, tgt[p]);
                CK(cudaMemcpyToSymbol(c_tgt, tgt, sizeof(tgt)));
                std::vector<Hit> h1, h2;
                launch(Bt, Ft, chunks - 1, 1, total, 0, h1);
                launch(Bt, Ft, chunks - 1, 1, total, skipDone, h2);
                int hitOff = 0; for (auto& h : h1) hitOff += mask_of(h) == planted;
                bool skipped = true; for (auto& h : h2) skipped &= mask_of(h) != planted;
                printf("  freeze self-test (k=%d, skip %s): %s  [%s]\n", nb + 1, skipDone ? "on" : "off",
                       (hitOff == 1 && skipped) ? "OK" : "FAILED", hitOff == 1 && skipped ? "found without the skip, absent with it" : "wrong");
                ok &= hitOff == 1 && skipped;
            }
            memcpy(tgt[p], saved, 20); CK(cudaMemcpyToSymbol(c_tgt, tgt, sizeof(tgt)));
            if (!ok) { fprintf(stderr, "freeze self-test FAILED - refusing to run\n"); return 1; }
        }

        // one database per puzzle: it stays on disk and the next search of this puzzle continues from it
        std::string dbPath = "freeze_masks_p" + std::to_string(p) + ".txt";
        std::set<u128> done;                                // masks of this (puzzle, m) already finished
        { std::ifstream db(dbPath); std::string ln;
          while (std::getline(db, ln)) {
              std::istringstream is(ln); int pp, mm; std::string hx;
              if (!(is >> pp >> mm >> hx) || pp != p || mm != m || hx.size() != 32) continue;
              done.insert(((u128)std::strtoull(hx.substr(0, 16).c_str(), nullptr, 16) << 64) | (u128)std::strtoull(hx.substr(16).c_str(), nullptr, 16));
          } }
        FILE* dbFile = fopen(dbPath.c_str(), "a");          // created on the first run of this puzzle
        auto db_add = [&](u128 mask) {
            if (dbFile) fprintf(dbFile, "%d %d %s\n", p, m, hexmask(mask).c_str());
        };
        auto exhausted = [&]() { return combos && (u128)done.size() >= combos; };
        auto sayExhausted = [&]() {
            status_end();
            printf("Freeze: all combinations were exhausted for puzzle %d with %d frozen ones (%s masks in %s) - nothing left to search\n",
                   p, m, u128_str(combos).c_str(), dbPath.c_str());
        };
        printf("mask list: %s, %zu of %s masks done for puzzle %d with %d frozen ones\n", dbPath.c_str(), done.size(),
               combos ? u128_str(combos).c_str() : "many", p, m);
        auto tstr = [](double s) {
            char t[32];
            if (s < 1e-3) snprintf(t, sizeof t, "%.1f us", s * 1e6);
            else if (s < 1.0) snprintf(t, sizeof t, "%.1f ms", s * 1e3);
            else if (s < 60.0) snprintf(t, sizeof t, "%.1f s", s);
            else if (s < 3600.0) snprintf(t, sizeof t, "%.1f min", s / 60.0);
            else if (s < 86400.0) snprintf(t, sizeof t, "%.1f h", s / 3600.0);
            else snprintf(t, sizeof t, "%.1f days", s / 86400.0);
            return std::string(t);
        };
        {   // the knob: --frz sets the lowest reachable ones count (k >= frz+1) and the mask size 2^(P-1-frz)
            printf("  --frz is the floor: this run reaches k = %d..%d, each mask is 2^%d = %s keys",
                   m + 1, nb + 1, f, u128_str(total).c_str());
            for (int d = 5; d <= 10 && m - d >= 1 && f + d <= 62; d += 5)
                printf(";  --frz %d -> 2^%d (%s per mask)", m - d, f + d, tstr((double)(((u128)1 << (f + d))) / 2.0e8).c_str());
            printf("\n");
        }

        u128 firstB = 0;                                    // --check only: m random ones of the known key
        if (known && chk) for (int c = 0; c < m; ) { int b = (int)(rng() % nb); u128 bit = bit128h(b); if (getbit(kk, b) && !(firstB & bit)) { firstB |= bit; c++; } }

        char b[512];
        u64 fastN = 0; double fastSum = 0, fastMin = 1e9, fastMax = 0;   // tiny masks: summarised, not one line each
        auto tFast = std::chrono::steady_clock::now();
        bool slowMask = est >= 0.2;                                      // per-mask line only when a mask is worth a line
        auto tRun = std::chrono::steady_clock::now();                    // для --seconds: общий лимит времени
        bool timeUp = false;
        for (u64 nMask = 1;; nMask++) {
            u128 B = 0;                                    // must start at 0: the picker tests !B
            if (known && chk) B = firstB;
            else {
                if (exhausted()) { sayExhausted(); return 0; }
                int tries = 0;
                for (; tries < 1000 && !B; tries++) { u128 v = rand_mask(); if (!done.count(v)) B = v; }
                if (!B) { status_end(); printf("Freeze: no unused mask after %d tries (%zu of %s done) - stopping\n", tries, done.size(), combos ? u128_str(combos).c_str() : "many"); return 0; }
            }
            u128 F = free_of(B);
            {   // invariant: exactly m ones, all inside the p-1 low bits (this is what caught the uninitialised-mask bug)
                int pc = (int)__builtin_popcountll((u64)B) + (int)__builtin_popcountll((u64)(B >> 64));
                if (pc != m || (B >> nb) != 0) {
                    status_end();
                    printf("Freeze: internal error - mask %s has %d ones inside %d low bits, expected %d\n",
                           bstr(B, nb).c_str(), pc, nb, m);
                    return 1;
                }
            }
            status_end();
            if (slowMask) {                                        // slow mask: the starting key in bits and hex, then what it costs
                printf("mask %llu: key %s\n"
                       "          (%d bits: the leftmost 1 is the always-set top bit, the other 1s are the %d frozen ones,\n"
                       "           0 = free place; the sweep starts with every free place 0)\n"
                       "          key hex %s   mask hex %s (database line)\n"
                       "          %s keys to sweep",
                       (unsigned long long)nMask, bstr(B | bit128h(p - 1), p).c_str(), p, m,
                       keyhex(B).c_str(), hexmask(B).c_str(), u128_str(total).c_str());
                double hc = hitChance((u64)done.size());
                printf(hc >= 0 ? ", key chance so far %.2f%%\n" : "\n", hc);
            }
            std::vector<Hit> hits;
            u64 first = 0;
            auto t0 = std::chrono::steady_clock::now(), lastRep = t0;
            while (first < chunks) {
                if (seconds > 0 && std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - tRun).count() >= seconds) {
                    timeUp = true; break;                                // --seconds: остановиться посреди маски
                }
                u64 cnt = std::min<u64>(chunks - first, (u64)NT);
                launch(B, F, first, cnt, total, skipDone, hits);
                first += cnt;
                u64 checked = std::min<u64>(total, first * CS);
                for (auto& h : hits) {
                    u128 key = key_of(mask_of(h)); u32 w[8]; key_words(key, w);
                    u32 hh[5]; key_h160(w, hh);
                    bool ok = !memcmp(hh, tgt[p], 20);
                    std::string rec = found_record(p, w, addr[p], ok);
                    if (!known) { FILE* f = fopen("found.txt", "a"); if (f) { fputs(rec.c_str(), f); fclose(f); } }
                    status_end();
                    printf("\n*** CANDIDATE in a frozen mask ***\n%s", rec.c_str()); fflush(stdout);
                    if (ok && known && chk) { printf("\n%s test hit on solved puzzle %d - freeze mode verified\n", "\x1b[92mOK\x1b[0m", p); return 0; }
                    if (ok && known) { printf("\n%s key of solved puzzle %d found in a random mask (no alarm: it is a solved one)\n", "\x1b[92mOK\x1b[0m", p); return 0; }
                    if (ok) { alarm(p, addr[p], false); return 0; }
                }
                auto now = std::chrono::steady_clock::now();
                double el = std::chrono::duration<double>(now - t0).count();
                if (std::chrono::duration<double>(now - lastRep).count() >= 10) {
                    lastRep = now;
                    snprintf(b, sizeof b, "frz %d mask %llu: %s / %s keys  %.2f%s  %.1f Mkeys/s", m, (unsigned long long)nMask,
                             u128_str((u128)checked).c_str(), u128_str(total).c_str(), 100.0 * (double)checked / (double)total, "%", (double)checked / el / 1e6);
                    status_printf("%s", b);                                                                  // one line, rewritten in place
                    if (load < 100) std::this_thread::sleep_for(std::chrono::duration<double>(1.0 * (100 - load) / load));
                }
            }
            double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            if (timeUp) {                                            // маска НЕ помечается пройденной
                status_end();
                printf("Freeze: --seconds %.0f reached at mask %llu (%s of %s keys); the mask is not recorded, "
                       "the next run starts it again\n",
                       seconds, (unsigned long long)nMask,
                       u128_str((u128)(first * CS)).c_str(), u128_str(total).c_str());
                return 0;
            }
            db_add(B); done.insert(B);
            if (el >= 0.2 || (known && chk)) {                       // one line per slow mask, with its real time
                status_end();
                printf("mask %llu: %s keys checked in %s - %s\n", (unsigned long long)nMask, u128_str(total).c_str(), tstr(el).c_str(),
                       (known && chk) ? "\x1b[91mNOT FOUND - the frozen ones were wrong\x1b[0m" : "\x1b[92mno key found\x1b[0m");
                if (dbFile) fflush(dbFile);
                fastN = 0; fastSum = 0; fastMin = 1e9; fastMax = 0; tFast = std::chrono::steady_clock::now();
                if (known && chk) return 1;
            } else {                                                 // tiny masks: min/mean/max once a second, no flood
                fastN++; fastSum += el; fastMin = std::min(fastMin, el); fastMax = std::max(fastMax, el);
                auto now = std::chrono::steady_clock::now();
                if (std::chrono::duration<double>(now - tFast).count() >= 1.0) {
                    status_end();
                    printf("frz %d mask %llu: %s keys, %s-%s (mean %s), %zu of %s done",
                           m, (unsigned long long)nMask, u128_str(total).c_str(), tstr(fastMin).c_str(), tstr(fastMax).c_str(),
                           tstr(fastSum / (double)fastN).c_str(), done.size(), combos ? u128_str(combos).c_str() : "many");
                    double hc = hitChance((u64)done.size());
                    printf(hc >= 0 ? ", key chance so far %.2f%%\n" : "\n", hc);
                    printf("  key %s  (%d bits: leftmost 1 is the top bit, other 1s are frozen, 0 = free)\n",
                           bstr(B | bit128h(p - 1), p).c_str(), p);
                    printf("  key hex %s   mask hex %s (database line)\n", keyhex(B).c_str(), hexmask(B).c_str());
                    if (dbFile) fflush(dbFile);
                    fastN = 0; fastSum = 0; fastMin = 1e9; fastMax = 0; tFast = now;
                }
            }
        }
    }

    // ---- --spread P: one puzzle, k ones at random places, the ones range kmin..kmax is split across threads ----
    if (spread) {
        int p = spread;
        if (p < 3 || p > 160) { fprintf(stderr, "--spread: puzzle 3..160\n"); return 1; }
        kmin = std::max(2, std::min(kmin, p - 1)); kmax = std::max(kmin, std::min(kmax, p - 1));   // k counts the top bit too
        bool known = false; for (auto& e : KNOWN) known |= e.p == p;
        u32 *dK; jac* dP; CK(cudaMalloc(&dK, NT * 20)); CK(cudaMalloc(&dP, NT * sizeof(jac)));
        std::vector<u32> hK(NT * 5, 0); std::vector<jac> hP(NT, to_jac(hpow2[p - 1]));
        for (size_t t = 0; t < NT; t++) { setbit(&hK[t * 5], p - 1, 1); hR[t] = rng() | 1; }
        CK(cudaMemcpy(dK, hK.data(), NT * 20, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dP, hP.data(), NT * sizeof(jac), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dR, hR.data(), NT * 8, cudaMemcpyHostToDevice)); CK(cudaMemset(dHitN, 0, 4));
        printf("%s", BANNER);
        printf("Spread: puzzle %d only%s, ones %d..%d split over %zu walkers, GPU load limit %d%%\n", p, known ? " (solved - test run)" : "", kmin, kmax, NT, load);
        if (watch) printf("Watch: walker #0 (k = %d), one step per frame; hash160 of every key is checked vs %s\n", kmin, addr[p].c_str());
        const double space = ldexp(1.0, p - 1);
        int nk = kmax - kmin + 1, steps = 1; double total = 0, paused = 0, lastTotal = 0, lastEl = 0;
        u32 W0[5] = {}; setbit(W0, p - 1, 1); long frame = 0;   // --watch: key of walker #0 before this frame
        // Live table (one row per ones count) goes straight to the terminal, not through tee into snake.log.
        // It needs 3 fixed rows (mantra, status, column header) plus one row per ones count and ~p+30 columns.
        // If the window is too short for all slices, the rows are paged instead of scrolling the window.
        FILE* tty = watch ? nullptr : fopen("/dev/tty", "w");
        int tcols = 100, trows = 24; tty_size(&tcols, &trows);
        const int perPage = std::max(1, trows - 4);   // 3 header rows + a spare row, or the last "\n" scrolls the window
        const bool table = tty && (perPage >= 3) && (p + 30 <= tcols);
        FILE* live = tty ? tty : stdout;   // one-line status stream: the tty keeps snake.log free of progress spam
        if (!table) printf("Window %dx%d is too small for the table (%d slices, needs ~%d cols): showing one status line instead\n", tcols, trows, nk, p + 30);
        else if (nk > perPage) printf("Window is %d rows tall, the table needs %d: %d slices per page, flipping every 15 s (a taller window shows all %d at once)\n", trows, nk + 4, perPage, nk);
        double rate = 0, lastDraw = -1; int temp = gpu_temp();
        auto start = std::chrono::steady_clock::now(), lastCool = start;
        for (;;) {
            auto t0 = std::chrono::steady_clock::now();
            k_spread<<<blocks, TPB>>>(p, kmin, nk, dK, dP, dR, steps, dHitN, dHits); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
            double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            total += (double)NT * steps; frame++;
            if (watch) {   // walker #0, exactly as the GPU left it: green 1 = just set, red 0 = just cleared
                u32 W[5]; CK(cudaMemcpy(W, dK, 20, cudaMemcpyDeviceToHost));
                std::string line; int on = 0;
                for (int i = p - 1; i >= 0; i--) {
                    int a = getbit(W0, i), b = getbit(W, i); on += b;
                    line += a == b ? (b ? "\x1b[97m1" : "\x1b[90m0") : (b ? "\x1b[92;1m1" : "\x1b[91;1m0");
                }
                u32 kk[8] = {}; memcpy(kk, W, 20);
                printf("\r%7ld  k=%2d %s\x1b[0m  %2d ones  %s\x1b[K",
                       frame, kmin, line.c_str(), on, key_str(kk).c_str() + 64 - (p + 3) / 4);
                memcpy(W0, W, 20); fflush(stdout);
                std::this_thread::sleep_for(std::chrono::milliseconds(120));
            }
            if (!watch && dt < 0.5 && steps < (1 << 20)) steps *= 2;   // watch: 1 step per frame
            if (load < 100) std::this_thread::sleep_for(std::chrono::duration<double>(dt * (100 - load) / load));
            std::vector<Hit> hits; read_hits(hits);
            for (auto& h : hits) {
                u32 kk[8] = {}, hh[5]; memcpy(kk, h.V, 20); key_h160(kk, hh);
                bool ok = !memcmp(hh, tgt[p], 20);
                std::string rec = found_record(p, kk, addr[p], ok);
                if (known) { printf("\n%s test hit after %.3e keys:\n%s", ok ? "\x1b[92mOK\x1b[0m" : "\x1b[91mFAILED\x1b[0m", total, rec.c_str()); return ok ? 0 : 1; }
                FILE* f = fopen("found.txt", "a"); if (f) { fputs(rec.c_str(), f); fclose(f); }
                printf("\n*** FOUND - written to found.txt ***\n%s", rec.c_str()); fflush(stdout);
                if (ok) { alarm(p, addr[p], false); return 0; }
            }
            auto now = std::chrono::steady_clock::now();
            double el = std::chrono::duration<double>(now - start).count() - paused;
            int e = (int)el;
            if (!watch && el - lastEl >= 10) {
                rate = (total - lastTotal) / (el - lastEl); lastTotal = total; lastEl = el; temp = gpu_temp();
                if (!table) {   // one line, rewritten in place (the table does not fit this window)
                    char lb[512]; snprintf(lb, sizeof lb, "[%d:%02d:%02d] %.3e keys | %.1f M/s | chance %.2e | expected %.3g years | %dC",
                                           e / 3600, e / 60 % 60, e % 60, total, rate / 1e6, total / space, space / rate / 3.156e7, temp);
                    live_write(live, tcols, lb);
                }
            }
            if (table && el - lastDraw >= 1) {   // whole table redrawn from the top of the window, about once a second
                lastDraw = el;
                double r = rate > 0 ? rate : total / el;
                std::vector<u32> G(nk * 5); CK(cudaMemcpy(G.data(), dK, nk * 20, cudaMemcpyDeviceToHost));   // thread j has k = kmin + j
                const int pages = (nk + perPage - 1) / perPage;
                const int page = pages > 1 ? (int)((long long)(el / 15) % pages) : 0;   // a short window pages through the slices
                const int j0 = page * perPage, j1 = std::min(nk, j0 + perPage);
                fprintf(tty, "\x1b[H\x1b[J%s", BANNER);
                char hd[512];
                snprintf(hd, sizeof hd, "Spread #%d  [%d:%02d:%02d]  %.3e keys | %.1f M/s | chance %.2e | %.3g years | %dC",
                         p, e / 3600, e / 60 % 60, e % 60, total, r / 1e6, total / space, space / r / 3.156e7, temp);
                if (pages > 1) snprintf(hd + strlen(hd), sizeof hd - strlen(hd), " | page %d/%d, ones %d..%d",
                                        page + 1, pages, kmin + j0, kmin + j1 - 1);
                if ((int)strlen(hd) > tcols - 1) hd[tcols - 1] = 0;   // never wrap a table row
                fprintf(tty, "%s\nones  %-*s  checked   share\n", hd, p, "current key of one walker (top bit first)");
                for (int j = j0; j < j1; j++) {
                    int k = kmin + j;
                    double keys = total * (double)(NT / nk + ((size_t)j < NT % nk)) / NT;
                    double slice = exp(lgamma(p) - lgamma(k) - lgamma(p - k + 1));   // C(p-1, k-1)
                    std::string bits; int last = -1;
                    for (int i = p - 1; i >= 0; i--) {
                        int b = getbit(&G[j * 5], i);
                        if (b != last) { bits += b ? "\x1b[97m" : "\x1b[90m"; last = b; }
                        bits += (char)('0' + b);
                    }
                    fprintf(tty, "%3d   %s\x1b[0m  %.2e  %.2e\n", k, bits.c_str(), keys, keys / slice);
                }
                fflush(tty);
            }
            if (seconds > 0 && el >= seconds) break;
            if (cool && std::chrono::duration<double>(now - lastCool).count() >= 50 * 60) {
                printf("\ncooling break 10 min...\n"); fflush(stdout);
                std::this_thread::sleep_for(std::chrono::minutes(10));
                paused += 600; lastCool = std::chrono::steady_clock::now();
            }
        }
        printf("\n"); return 0;
    }

    // ---- search ----
    u8 act[161] = {}; int nAct = 0;
    for (int p = 71; p <= 160; p++) { act[p] = 1; for (int s : SOLVED) if (s == p) act[p] = 0; nAct += act[p]; }
    CK(cudaMemcpyToSymbol(c_active, act, sizeof(act)));
    reset_threads(); upload();
    if (show > 0) {  // print snake #0's keys exactly as the GPU walks them, 160-bit binary
        for (int st = 0; st < show; st++) {
            u32 V[5]; CK(cudaMemcpy(V, dV, 20, cudaMemcpyDeviceToHost));
            printf("--- snake step %d ---\n", st);
            for (int p = 71; p <= 160; p++) {
                u32 k[8]; V_to_key(V, p, k);
                char b[161]; for (int i = 0; i < 160; i++) b[i] = '0' + getbit(k, 159 - i); b[160] = 0;
                printf("P%03d %s%s\n", p, b, act[p] ? "" : "  (solved, skipped)");
            }
            k_snake<<<blocks, TPB>>>(dV, dR, 1, dHitN, dHits); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        }
        return 0;
    }
    bool limitOn = load < 100; if (!limitOn) load = 70;
    auto load_str = [&]() { return limitOn ? std::to_string(load) + "%" : std::string("off"); };
    printf("%s", BANNER);
    printf("Snake: %d unsolved puzzles in 71..160, %zu walkers, GPU load limit %s\n", nAct, NT, load_str().c_str());
    kb_init();
    if (kb_raw) printf("Keys: L = load limit on/off, + / - = change the limit by 10%%\n");
    double weight = 0;   // chance per snake shift of any hit, in units of 2^-70
    for (int p = 71; p <= 160; p++) if (act[p]) weight += ldexp(1.0, 71 - p);
    int steps = 1; double total = 0, shifts = 0, paused = 0, lastTotal = 0, lastEl = 0; bool first = true;
    // --watch: эталон — известный ключ решённого пазла `ref`; смотрим, насколько близко подходит кандидат (наглядно, медленно).
    const char* refhex = nullptr; for (auto& e : KNOWN) if (e.p == ref) refhex = e.key;
    if (watch && !refhex) { fprintf(stderr, "--ref %d: нет известного ключа (эталоном может быть только решённый пазл)\n", ref); return 1; }
    u32 kref[8]; std::string refStr; int refOnes = 0, rec70 = 0;
    if (watch) {
        key_hex(refhex, kref);
        for (int i = ref - 1; i >= 0; i--) { int b = getbit(kref, i); refStr += (char)('0' + b); refOnes += b; }
        printf("watch: кандидаты против ключа пазла %d (%d единиц из %d)\n       %s\n", ref, refOnes, ref, refStr.c_str());
        if (ones >= 0) printf("       режим: нулевая строка + %d единиц в случайных позициях\n", ones);
        if (ones >= 0 && ones != refOnes)
            printf("       \x1b[93mвнимание: у ключа %d единиц — при --ones %d полного совпадения не будет (нужно --ones %d)\x1b[0m\n", refOnes, ones, refOnes);
    }
    const int WATCH_TRIES = 8000;   // кандидатов за кадр в watch: маленький пазл реально решается на глазах за секунды
    auto start = std::chrono::steady_clock::now(), lastPrint = start, lastCool = start;
    for (;;) {
        auto t0 = std::chrono::steady_clock::now();
        k_snake<<<blocks, TPB>>>(dV, dR, steps, dHitN, dHits); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        total += (double)NT * steps * nAct; shifts += (double)NT * steps;
        if (!watch && dt < 0.5 && steps < (1 << 20)) steps *= 2;   // aim ~0.5-1 s per launch (в watch держим steps=1 — видно каждый сдвиг)
        for (int c; (c = kb_get()) != -1;) {
            if (c == 'l' || c == 'L') limitOn = !limitOn;
            else if (c == '+' || c == '=') { load = std::min(90, load + 10); limitOn = true; }
            else if (c == '-' || c == '_') { load = std::max(10, load - 10); limitOn = true; }
            else continue;
            status_end();
            printf("GPU load limit: %s\n", load_str().c_str()); fflush(stdout);
        }
        if (limitOn) std::this_thread::sleep_for(std::chrono::duration<double>(dt * (100 - load) / load));   // duty cycle
        std::vector<Hit> hits; read_hits(hits);
        int foundP = 0;
        for (auto& h : hits) {   // write every hit first, then stop
            u32 k[8], hh[5]; V_to_key(h.V, h.p, k); key_h160(k, hh);
            bool ok = !memcmp(hh, tgt[h.p], 20);
            std::string rec = found_record(h.p, k, addr[h.p], ok);
            FILE* f = fopen("found.txt", "a"); if (f) { fputs(rec.c_str(), f); fclose(f); }
            char cwd[512]; if (!getcwd(cwd, sizeof cwd)) strcpy(cwd, ".");
            printf("\n*** FOUND - written to %s/found.txt ***\n%s", cwd, rec.c_str()); fflush(stdout);
            if (ok && !foundP) foundP = h.p;
        }
        if (foundP) { alarm(foundP, addr[foundP], false); return 0; }
        if (watch) {   // визуализация: совпадения кандидата с известным ключом пазла ref; показываем лучшего за кадр
            char best[161]; int tot = 0;
            if (ones >= 0) {              // кандидаты: нули длины ref + `ones` единиц в случайных позициях, пачка за кадр
                int o = ones > ref ? ref : ones;
                for (int t = 0; t < WATCH_TRIES; t++) {
                    char b0[161]; for (int i = 0; i < ref; i++) b0[i] = '0';
                    int idx[160]; for (int i = 0; i < ref; i++) idx[i] = i;
                    for (int i = 0; i < o; i++) { int j = i + rng() % (ref - i); int z = idx[i]; idx[i] = idx[j]; idx[j] = z; b0[idx[i]] = '1'; }
                    int m = 0; for (int i = 0; i < ref; i++) m += b0[i] == refStr[i];
                    if (m > tot) { tot = m; for (int i = 0; i < ref; i++) best[i] = b0[i]; }
                    if (m == ref) break;   // полное совпадение — пазл «решён» на глазах
                }
            } else {                      // кандидат: ключ змейки #0 для пазла ref (сдвиговый регистр)
                u32 V0[5]; CK(cudaMemcpy(V0, dV, 20, cudaMemcpyDeviceToHost));
                u32 k0[8]; V_to_key(V0, ref, k0);
                for (int i = 0; i < ref; i++) { best[i] = '0' + getbit(k0, ref - 1 - i); tot += best[i] == refStr[i]; }
            }
            std::string line;   // совпавший бит горит зелёным, несовпавший — серая точка
            for (int i = 0; i < ref; i++) { if (best[i] == refStr[i]) { line += "\x1b[92m"; line += best[i]; } else line += "\x1b[90m."; }
            line += "\x1b[0m";
            if (tot > rec70) rec70 = tot;
            printf("\r%s  %d/%d  рекорд %d", line.c_str(), tot, ref, rec70);
            fflush(stdout);
            if (tot == ref) { printf("\n\x1b[92m*** РЕШЕНО! пазл %d найден на глазах: %s ***\x1b[0m\a\n", ref, refStr.c_str()); return 0; }
            std::this_thread::sleep_for(std::chrono::milliseconds(80));
        }
        auto now = std::chrono::steady_clock::now();
        double el = std::chrono::duration<double>(now - start).count() - paused;
        if (std::chrono::duration<double>(now - lastPrint).count() >= 10) {
            lastPrint = now;
            const double TWO70 = 1.1805916207174113e21;
            int e = (int)el;
            if (first) {
                first = false;
                double years = TWO70 / weight / (shifts / el) / 3.156e7;
                status_end();
                printf("Expected time to any hit at this speed: %.1f million years (%.0f snake shifts/s)\n", years / 1e6, shifts / el);
            }
            double now_rate = (total - lastTotal) / (el - lastEl) / 1e6; lastTotal = total; lastEl = el;
            status_printf("[%d:%02d:%02d] keys %.3e | %.1f M/s (avg %.1f) | chance %.2e | %dC | load %s",
                   e / 3600, e / 60 % 60, e % 60, total, now_rate, total / el / 1e6, shifts * weight / TWO70, gpu_temp(), load_str().c_str());
        }
        if (seconds > 0 && el >= seconds) break;
        if (cool && std::chrono::duration<double>(now - lastCool).count() >= 50 * 60) {
            status_end();
            printf("cooling break 10 min...\n"); fflush(stdout);
            std::this_thread::sleep_for(std::chrono::minutes(10));
            paused += 600; lastCool = std::chrono::steady_clock::now();
        }
    }
    {   // final line, printed once the loop is over (the live line is wiped first)
        double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() - paused;
        int e = (int)el;
        status_end();
        printf("Stopped: [%d:%02d:%02d] keys %.3e | avg %.1f Mkeys/s | snake shifts %.3e | hits 0\n",
               e / 3600, e / 60 % 60, e % 60, total, total / el / 1e6, shifts);
        fflush(stdout);
    }
    return 0;
}
