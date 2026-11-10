/*
 * gen_graph.c -- write a synthetic symmetric Matrix Market graph for testing.
 *
 * Usage:   ./gen_graph n edges hub_fraction seed > graph.mtx
 * Example: ./gen_graph 2000000 8000000 0.2 1 > synth.mtx
 *
 * A fraction hub_fraction of the edges touches one of 4 "hub" vertices;
 * the rest join uniformly random vertices. This mimics the structure of
 * the MAWI traffic graphs: most rows are short, a few rows are huge.
 * Only the lower triangle is written (symmetric format), with values in (0,1].
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

static uint64_t rng_state;
static uint64_t rng(void)                 /* xorshift64*: fast, good enough */
{
    rng_state ^= rng_state >> 12;
    rng_state ^= rng_state << 25;
    rng_state ^= rng_state >> 27;
    return rng_state * 2685821657736338717ULL;
}

int main(int argc, char **argv)
{
    if (argc != 5) {
        fprintf(stderr, "usage: %s n edges hub_fraction seed\n", argv[0]);
        return 1;
    }
    long long n = atoll(argv[1]), e = atoll(argv[2]);
    double hub = atof(argv[3]);
    rng_state = 88172645463325252ULL ^ (uint64_t)atoll(argv[4]);

    printf("%%%%MatrixMarket matrix coordinate real symmetric\n");
    printf("%% synthetic graph: %lld vertices, hub fraction %.2f\n", n, hub);
    printf("%lld %lld %lld\n", n, n, e);
    for (long long k = 0; k < e; k++) {
        long long i, j;
        if ((double)(rng() >> 11) / 9007199254740992.0 < hub)
            i = (long long)(rng() % 4);                 /* one of 4 hubs */
        else
            i = (long long)(rng() % (uint64_t)n);
        j = (long long)(rng() % (uint64_t)n);
        if (i < j) { long long tmp = i; i = j; j = tmp; } /* lower triangle */
        double v = (double)((rng() >> 11) % 1000 + 1) / 1000.0;
        printf("%lld %lld %.3f\n", i + 1, j + 1, v);
    }
    return 0;
}
