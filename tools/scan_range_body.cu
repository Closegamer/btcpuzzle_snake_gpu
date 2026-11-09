// scan_range: реальная проверка ПРОИЗВОЛЬНОГО диапазона ключей (не по углу, как пушка).
// Полезна, когда известный ключ надо не «искать веером», а проверять точные кандидаты:
//   key_i = start + i,  i = 0..count-1
// Каждый поток берёт свой непрерывный кусок и идёт по нему прибавлением G (одно сложение
// точки на ключ), инверсия — батчами по SR_B (измерено 2026-11-08: батч 16 даёт 368 Мключ/с,
// 64 — заметно больше, потому что инверсия при батче 16 съедала ~60 % времени).
#ifndef SR_B
#define SR_B 128             // батч инверсии (было 16: 368 Мключ/с; 64 -> 502; 128 -> 563; 256 -> 569)
#endif

__constant__ aff c_p2[161];      // 2^b * G
__constant__ u32 c_tgt[5];       // hash160 цели (слова RIPEMD, как в snake.cu)

__device__ void flush_scan(fe* X, fe* Y, fe* Z, fe* acc, u64* offs, int n,
                           u64* hitOff, u32* hitN) {
    acc[0] = Z[0];
    for (int i = 1; i < n; i++) fe_mul(acc[i], acc[i - 1], Z[i]);
    fe inv; fe_inv(inv, acc[n - 1]);
    for (int i = n - 1; i >= 0; i--) {
        fe zi, zi2, x, y;
        if (i) { fe_mul(zi, inv, acc[i - 1]); fe_mul(inv, inv, Z[i]); } else zi = inv;
        fe_sqr(zi2, zi); fe_mul(x, X[i], zi2); fe_mul(y, Y[i], zi2); fe_mul(y, y, zi);
        u32 h[5];
        hash160_pub(x, y.v[0] & 1, h);
        if (h[0] == c_tgt[0] && h[1] == c_tgt[1] && h[2] == c_tgt[2] &&
            h[3] == c_tgt[3] && h[4] == c_tgt[4]) {
            u32 k = atomicAdd(hitN, 1);
            if (k < 64) hitOff[k] = offs[i];
        }
    }
}

// Safe add for the scanner: start can be a power of two, so P0 == c_p2[b] and the
// ordinary madd() (distinct points only) would divide by zero. Also handles infinity.
HD void madd_scan(jac& P, const aff& Q) {
    fe zero = {};
    if (fe_eq(P.Z, zero)) {
        fe one = {{1, 0, 0, 0, 0, 0, 0, 0}};
        P.X = Q.x; P.Y = Q.y; P.Z = one; return;
    }
    fe z1z1, u2, s2, h, r;
    fe_sqr(z1z1, P.Z); fe_mul(u2, Q.x, z1z1);
    fe_mul(s2, Q.y, P.Z); fe_mul(s2, s2, z1z1);
    fe_sub(h, u2, P.X); fe_sub(r, s2, P.Y);
    if (fe_eq(h, zero)) {
        if (fe_eq(r, zero)) jdbl(P);                 // P == Q
        else { P.X = zero; P.Y = zero; P.Z = zero; } // P == -Q -> infinity
        return;
    }
    madd(P, Q);
}

__global__ void __launch_bounds__(256) k_scan(jac P0, u64 per, u64 total,
                                              u64* hitOff, u32* hitN) {
    u64 t = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    u64 base = t * per;
    if (base >= total) return;
    u64 cnt = total - base;
    if (cnt > per) cnt = per;
    jac P = P0;                                   // start * G
    for (int b = 0; b < 64; b++)                  // + base * G
        if ((base >> b) & 1ull) { aff g = c_p2[b]; madd_scan(P, g); }
    fe X[SR_B], Y[SR_B], Z[SR_B], acc[SR_B];
    u64 offs[SR_B];
    int n = 0;
    for (u64 j = 0; j < cnt; j++) {
        if (j) { aff g = c_p2[0]; madd_scan(P, g); }   // + G
        X[n] = P.X; Y[n] = P.Y; Z[n] = P.Z; offs[n] = base + j;
        if (++n == SR_B) { flush_scan(X, Y, Z, acc, offs, n, hitOff, hitN); n = 0; }
    }
    if (n) flush_scan(X, Y, Z, acc, offs, n, hitOff, hitN);
}

// ---------------------------------------------------------------- host
static std::string u128_dec(u128 v) {
    if (!v) return "0";
    std::string s;
    while (v) { s.insert(s.begin(), char('0' + (int)(v % 10))); v /= 10; }
    return s;
}
static std::string u128_hex(u128 v) {
    char b[40];
    snprintf(b, sizeof b, "%016llx%016llx", (unsigned long long)(v >> 64), (unsigned long long)v);
    return b;
}
// Строгий разбор: только десятичные цифры (или 0x + hex). Строка вида «1.9e+21»
// раньше молча превращалась в мусор — теперь это ошибка.
static bool parse_u128(const char* s, u128& out) {
    // Десятичное или 0x-шестнадцатеричное, строго до 128 бит. Раньше hex шёл через
    // strtoull: всё, что длиннее 64 бит, молча превращалось в 0xFFFF…FFFF (обрезка),
    // т.е. --start/--plant могли проверять совсем не тот диапазон.
    int base = 10;
    const char* q = s;
    if (s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) { base = 16; q = s + 2; }
    if (!*q) return false;
    u128 v = 0;
    for (const char* t = q; *t; t++) {
        int d;
        if (*t >= '0' && *t <= '9') d = *t - '0';
        else if (base == 16 && *t >= 'a' && *t <= 'f') d = *t - 'a' + 10;
        else if (base == 16 && *t >= 'A' && *t <= 'F') d = *t - 'A' + 10;
        else return false;
        if (d >= base) return false;
        if (v > (~(u128)0 - (u128)d) / (u128)base) return false;   // переполнение 128 бит
        v = v * (u128)base + (u128)d;
    }
    out = v;
    return true;
}

int main(int argc, char** argv) {
    u128 start = (u128)1 << 70;
    u64 count = 1000000, per = 0;
    int blocks = 1024;
    const char* target = nullptr;
    const char* plant = nullptr;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--start") && i + 1 < argc) {
            if (!parse_u128(argv[++i], start)) {
                printf("--start: ожидается десятичное число (или 0x hex), получено '%s'\n",
                       argv[i]);
                return 1;
            }
        }
        else if (!strcmp(argv[i], "--count") && i + 1 < argc) count = strtoull(argv[++i], nullptr, 10);
        else if (!strcmp(argv[i], "--per") && i + 1 < argc) per = strtoull(argv[++i], nullptr, 10);
        else if (!strcmp(argv[i], "--blocks") && i + 1 < argc) blocks = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--target") && i + 1 < argc) target = argv[++i];
        else if (!strcmp(argv[i], "--plant") && i + 1 < argc) plant = argv[++i];
        else {
            printf("usage: scan_range [--start KEY] [--count N] [--blocks B] [--per K]\n"
                   "                  [--target ADDR|hash160hex] [--plant KEY]\n"
                   "  ключи: start .. start+count-1 (старт по умолчанию 2^70)\n"
                   "  --plant K  — цель := hash160(K), чтобы проверить, что сканер находит\n");
            return 1;
        }
    }

    if (blocks <= 0) { printf("--blocks: must be a positive number (got %d)\n", blocks); return 1; }

    hpow2[0] = {fe_hex("79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798"),
                fe_hex("483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8")};
    for (int k = 1; k <= 160; k++) { jac Pp = to_jac(hpow2[k - 1]); jdbl(Pp); hpow2[k] = to_aff(Pp); }
    CK(cudaMemcpyToSymbol(c_p2, hpow2, sizeof(hpow2)));

    u32 tgt[5] = {0, 0, 0, 0, 0};
    if (plant) {
        u32 kk[8] = {0};
        u128 pk = 0;
        if (!parse_u128(plant, pk)) { printf("--plant: плохое число '%s'\n", plant); return 1; }
        for (int w = 0; w < 4; w++) kk[w] = (u32)(pk >> (32 * w));
        key_h160(kk, tgt);
        printf("plant: ключ %s (hex %s) -> hash160 ", u128_dec(pk).c_str(), u128_hex(pk).c_str());
        for (int i = 0; i < 5; i++) printf("%08x", tgt[i]);
        printf("\n");
    } else if (target) {
        std::string ts(target);
        while (!ts.empty() && (ts.back() == '\r' || ts.back() == '\n' ||
                               ts.back() == ' ' || ts.back() == '\t')) ts.pop_back();
        target = ts.c_str();
        if (ts.size() == 40) {
            for (int i = 0; i < 5; i++) {
                char b[9] = {0};
                memcpy(b, ts.c_str() + 8 * i, 8);
                tgt[i] = (u32)strtoul(b, nullptr, 16);
            }
        } else if (!b58_h160(ts, tgt)) {
            printf("не разобрал --target (ни адрес, ни 40 hex): '%s'\n", ts.c_str());
            return 1;
        }
    } else {
        printf("нужен --target или --plant\n");
        return 1;
    }
    CK(cudaMemcpyToSymbol(c_tgt, tgt, sizeof(tgt)));

    u32 sk[8] = {0};
    for (int w = 0; w < 4; w++) sk[w] = (u32)(start >> (32 * w));
    jac P0 = point_of_key(sk);

    u64 threads = (u64)blocks * 256;
    if (!per) per = (count + threads - 1) / threads;
    if (!per) per = 1;
    u64* dOff; u32* dHitN;
    u64* hOff = (u64*)calloc(64, sizeof(u64));
    CK(cudaMalloc(&dOff, 64 * sizeof(u64)));
    CK(cudaMalloc(&dHitN, 4));
    CK(cudaMemset(dHitN, 0, 4));

    printf("scan: start=%s (hex %s), count=%llu, blocks=%d, per-thread=%llu\n",
           u128_dec(start).c_str(), u128_hex(start).c_str(),
           (unsigned long long)count, blocks, (unsigned long long)per);
    CK(cudaDeviceSynchronize());
    auto t0 = std::chrono::steady_clock::now();
    k_scan<<<blocks, 256>>>(P0, per, count, dOff, dHitN);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    u32 nH = 0;
    CK(cudaMemcpy(&nH, dHitN, 4, cudaMemcpyDeviceToHost));
    if (nH) {
        CK(cudaMemcpy(hOff, dOff, 64 * sizeof(u64), cudaMemcpyDeviceToHost));
        printf("\n*** НАЙДЕНО %u совпадений:\n", nH);
        for (u32 i = 0; i < nH && i < 64; i++) {
            u128 k = start + (u128)hOff[i];
            printf("  ключ %s  (hex %s)  offset %llu\n", u128_dec(k).c_str(),
                   u128_hex(k).c_str(), (unsigned long long)hOff[i]);
        }
    }
    printf("проверено %llu ключей за %.2f s = %.1f Мключ/с, совпадений %u\n",
           (unsigned long long)count, dt, (double)count / dt / 1e6, nH);
    return nH ? 0 : 1;
}
