#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <immintrin.h>

#define DNA_SIZE_MB 256
#define DNA_SIZE ((size_t)DNA_SIZE_MB * 1024 * 1024)
#define NUM_THREADS 4

long long global_A = 0;
long long global_C = 0;
long long global_G = 0;
long long global_T = 0;

pthread_mutex_t mutex;

typedef struct {
    const char *buf;
    size_t start;
    size_t end;
} thread_arg;

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void generate_dna(char *buf, size_t size)
{
    static const char nucleotides[4] = {'A', 'C', 'G', 'T'};
    srand(42);

    for (size_t i = 0; i < size; i++)
        buf[i] = nucleotides[rand() % 4];
}

static void count_scalar(const char *buf, size_t size, long long *count_A, long long *count_C, long long *count_G, long long *count_T)
{
    long long a = 0, c = 0, g = 0, t = 0;
    for(size_t i = 0; i < size; i++){
        switch (buf[i]) {
            case 'A': 
		a++; 
		break;
            case 'C': 
	 	c++;
	       	break;
            case 'G':
	       	g++; 
		break;
            case 'T':
	       	t++; 
		break;
        }
    }
    *count_A = a;
    *count_C = c;
    *count_G = g;
    *count_T = t;
}

static void count_simd(const char *buf, size_t size, long long *count_A, long long *count_C, long long *count_G, long long *count_T)
{
    long long a = 0, c = 0, g = 0, t = 0;

    __m256i vec_A = _mm256_set1_epi8('A');
    __m256i vec_C = _mm256_set1_epi8('C');
    __m256i vec_G = _mm256_set1_epi8('G');
    __m256i vec_T = _mm256_set1_epi8('T');

    size_t i = 0;
    for (; i + 32 <= size; i += 32) {
        __m256i chunk = _mm256_loadu_si256((__m256i *)(buf + i));
        a += __builtin_popcount(_mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, vec_A)));
        c += __builtin_popcount(_mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, vec_C)));
        g += __builtin_popcount(_mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, vec_G)));
        t += __builtin_popcount(_mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, vec_T)));
    }

    for (; i < size; i++) {
        switch (buf[i]) {
            case 'A': 
		a++; 
		break;
            case 'C': 
		c++; 
		break;
            case 'G': 
		g++; 
		break;
            case 'T': 
		t++; 
		break;
        }
    }
    *count_A = a;
    *count_C = c;
    *count_G = g;
    *count_T = t;
}

static void *thread_func(void *arg)
{
    thread_arg *th = (thread_arg *)arg;

    long long a = 0, c = 0, g = 0, t = 0;
    for (size_t i = th->start; i < th->end; i++) {
        switch (th->buf[i]) {
            case 'A': 
	 	a++; 
		break;
            case 'C':
	       	c++; 
		break;
            case 'G': 
		g++; 
		break;
            case 'T': 
		t++; 
		break;
        }
    }
    pthread_mutex_lock(&mutex);
    global_A += a;
    global_C += c;
    global_G += g;
    global_T += t;
    pthread_mutex_unlock(&mutex);

    return NULL;
}

static void *thread_func_simd(void *arg)
{
    thread_arg *th = (thread_arg *)arg;

    long long a = 0, c = 0, g = 0, t = 0;

    __m256i vec_A = _mm256_set1_epi8('A');
    __m256i vec_C = _mm256_set1_epi8('C');
    __m256i vec_G = _mm256_set1_epi8('G');
    __m256i vec_T = _mm256_set1_epi8('T');

    size_t i = th->start;
    for(; i + 32 <= th->end; i += 32){
        __m256i chunk = _mm256_loadu_si256((__m256i *)(th->buf + i));
        a += __builtin_popcount(_mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, vec_A)));
        c += __builtin_popcount(_mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, vec_C)));
        g += __builtin_popcount(_mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, vec_G)));
        t += __builtin_popcount(_mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, vec_T)));
    }

    for(; i < th->end; i++){
        switch (th->buf[i]) {
            case 'A': 
		a++; 
		break;
            case 'C':
	       	c++; 
		break;
            case 'G': 
		g++; 
		break;
            case 'T': 
		t++; 
		break;
        }
    }
    pthread_mutex_lock(&mutex);
    global_A += a;
    global_C += c;
    global_G += g;
    global_T += t;
    pthread_mutex_unlock(&mutex);

    return NULL;
}

int main()
{
    char *dna = malloc(DNA_SIZE);
    if (!dna) {
        printf("malloc failed\n");
        exit(1);
    }

    pthread_mutex_init(&mutex, NULL);

    printf("Generating random DNA.\n");
    generate_dna(dna, DNA_SIZE);
    printf("Finished.\n");

    long long cA, cC, cG, cT;
    double start_sec = now_sec();
    count_scalar(dna, DNA_SIZE, &cA, &cC, &cG, &cT);
    double time_diff = now_sec() - start_sec;

    printf("DNA size: %d MB\n", DNA_SIZE_MB);
    printf("Threads used: %d\n", NUM_THREADS);
    printf("Counts (A C G T):\n");
    printf("%lld %lld %lld %lld\n", cA, cC, cG, cT);
    printf("Scalar time: %.3f sec\n\n", time_diff);

    
    global_A = global_C = global_G = global_T = 0;
    
    pthread_t threads[NUM_THREADS];
    thread_arg args[NUM_THREADS];
    size_t chunk = DNA_SIZE / NUM_THREADS;

    start_sec = now_sec();
    for(int i = 0; i < NUM_THREADS; i++) {
        args[i].buf = dna;
        args[i].start = i * chunk;
        args[i].end = (i == NUM_THREADS - 1) ? DNA_SIZE : (i + 1) * chunk;
        if(pthread_create(&threads[i], NULL, thread_func, &args[i]) != 0){
            printf("thread create failed\n");
            exit(1);
        }
    }
    for(int i = 0; i < NUM_THREADS; i++) {
        if(pthread_join(threads[i], NULL) != 0) {
            printf("thread join failed\n");
            exit(1);
        }
    }
    time_diff = now_sec() - start_sec;
    printf("Multithreading time: %.3f sec\n", time_diff);
    if(cA == global_A && cC == global_C && cG == global_G && cT == global_T)
        printf("Multithreading correctness check: PASSED\n\n");
    else
        printf("Multithreading correctness check: FAILED\n\n");

    
    global_A = global_C = global_G = global_T = 0;
    
    start_sec = now_sec();
    count_simd(dna, DNA_SIZE, &global_A, &global_C, &global_G, &global_T);
    time_diff = now_sec() - start_sec;

    printf("SIMD time: %.3f sec\n", time_diff);
    if(cA == global_A && cC == global_C && cG == global_G && cT == global_T)
        printf("SIMD correctness check: PASSED\n\n");
    else
        printf("SIMD correctness check: FAILED\n\n");


    global_A = global_C = global_G = global_T = 0;
   
    start_sec = now_sec();
    for(int i = 0; i < NUM_THREADS; i++) {
    	args[i].buf = dna;
    	args[i].start = i * chunk;
    	args[i].end = (i == NUM_THREADS - 1) ? DNA_SIZE : (i + 1) * chunk;
    	if(pthread_create(&threads[i], NULL, thread_func_simd, &args[i]) != 0) {
        	printf("thread create failed\n");
        	exit(1);
	}
    }
    for(int i = 0; i < NUM_THREADS; i++){
        if(pthread_join(threads[i], NULL) != 0) {
            printf("thread join failed\n");
            exit(1);
	}
    }
    time_diff = now_sec() - start_sec;

    printf("SIMD + Multithreading time: %.3f sec\n", time_diff);
    if(cA == global_A && cC == global_C && cG == global_G && cT == global_T)
        printf("SIMD + Multithreading correctness check: PASSED\n");
    else
        printf("SIMD + Multithreading correctness check: FAILED\n");

    pthread_mutex_destroy(&mutex);
    free(dna);
    return 0;
}
