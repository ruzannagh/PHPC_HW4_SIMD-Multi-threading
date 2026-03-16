#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <time.h>
#include <immintrin.h>

#define NUM_THREADS 4
#define DNA_SIZE_MB 256
#define DNA_SIZE (DNA_SIZE_MB * 1024 * 1024)

pthread_mutex_t mutex;

char *dna;
long counts[4];
long partial_counts[NUM_THREADS][4];

void generate_dna() 
{
    dna = (char *)malloc(DNA_SIZE);
    if(dna == NULL){
        printf("Malloc failed\n");
        exit(1);
    }

    for(int i = 0; i < DNA_SIZE; i++)
        dna[i] = "ACGT"[rand() % 4];
}

void count_scalar(long *counts) 
{
    for (int i = 0; i < DNA_SIZE; i++){
        if(dna[i] == 'A') 
		counts[0]++;
        else if(dna[i] == 'C') 
		counts[1]++;
        else if(dna[i] == 'G') 
		counts[2]++;
        else if(dna[i] == 'T') 
		counts[3]++;
    }
}

void *thread_worker(void *arg) 
{
    int id = *((int *)arg);
    int start = id * (DNA_SIZE / NUM_THREADS);
    int end = (id == NUM_THREADS - 1) ? DNA_SIZE : (id + 1) * (DNA_SIZE / NUM_THREADS);

    long a = 0, c = 0, g = 0, t = 0;
    for(int i = start; i < end; i++){
        if(dna[i] == 'A')
		a++;
        else if(dna[i] == 'C') 
		c++;
        else if(dna[i] == 'G') 
		g++;
        else if(dna[i] == 'T') 
		t++;
    }

    pthread_mutex_lock(&mutex);
    counts[0] += a;
    counts[1] += c;
    counts[2] += g;
    counts[3] += t;
    pthread_mutex_unlock(&mutex);

    return NULL;
}

void count_simd() 
{
    long a = 0, c = 0, g = 0, t = 0;

    __m256i va = _mm256_set1_epi8('A');
    __m256i vc = _mm256_set1_epi8('C');
    __m256i vg = _mm256_set1_epi8('G');
    __m256i vt = _mm256_set1_epi8('T');

    for (int i = 0; i + 32 <= DNA_SIZE; i += 32){
        __m256i chunk = _mm256_loadu_si256((__m256i *)(dna + i));

        a += __builtin_popcount(_mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, va)));
        c += __builtin_popcount(_mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, vc)));
        g += __builtin_popcount(_mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, vg)));
        t += __builtin_popcount(_mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, vt)));
    }

    for (int i = 0; i < DNA_SIZE; i++){
        if (dna[i] == 'A') 
		a++;
        else if(dna[i] == 'C') 
		c++;
        else if(dna[i] == 'G') 
		g++;
        else if(dna[i] == 'T') 
		t++;
    }

    counts[0] = a; counts[1] = c; counts[2] = g; counts[3] = t;
}

void *thread_simd_worker(void *arg) 
{
    int id = *((int *)arg);
    int start = id * (DNA_SIZE / NUM_THREADS);
    int end = (id == NUM_THREADS - 1) ? DNA_SIZE : (id + 1) * (DNA_SIZE / NUM_THREADS);

    long a = 0, c = 0, g = 0, t = 0;

    __m256i va = _mm256_set1_epi8('A');
    __m256i vc = _mm256_set1_epi8('C');
    __m256i vg = _mm256_set1_epi8('G');
    __m256i vt = _mm256_set1_epi8('T');

    for (int i = start; i + 32 <= end; i += 32){
        __m256i chunk = _mm256_loadu_si256((__m256i *)(dna + i));

        a += __builtin_popcount(_mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, va)));
        c += __builtin_popcount(_mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, vc)));
        g += __builtin_popcount(_mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, vg)));
        t += __builtin_popcount(_mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, vt)));
    }

    for (int i = start; i < end; i++){
        if (dna[i] == 'A')
	        a++;
        else if (dna[i] == 'C') 
		c++;
        else if (dna[i] == 'G') 
		g++;
        else if (dna[i] == 'T')
	       	t++;
    }

    partial_counts[id][0] = a;
    partial_counts[id][1] = c;
    partial_counts[id][2] = g;
    partial_counts[id][3] = t;

    return NULL;
}

int main() 
{
    srand(42);
    generate_dna();

    printf("DNA size: %d MB\n", DNA_SIZE_MB);
    printf("Threads used: %d\n", NUM_THREADS);

    pthread_t threads[NUM_THREADS];
    int thread_ids[NUM_THREADS];
    pthread_mutex_init(&mutex, NULL);

    struct timespec start, end;
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    count_scalar(counts);
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("Counts (A C G T): %ld %ld %ld %ld\n", counts[0], counts[1], counts[2], counts[3]);
    //printf("Scalar time: %.3f sec\n", (double)(end - start) / CLOCKS_PER_SEC);
    printf("Scalar time: %.3f sec\n", (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) * 1e-9);

    for(int i = 0; i < 4; ++i)
	counts[i] = 0;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < NUM_THREADS; i++) {
        thread_ids[i] = i;
        if (pthread_create(&threads[i], NULL, thread_worker, &thread_ids[i]) != 0){
            printf("thread create failed\n");
            exit(1);
        }
    }
    for (int i = 0; i < NUM_THREADS; i++){
        if (pthread_join(threads[i], NULL) != 0){
            printf("thread join failed\n");
            exit(1);
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("Multithreading time: %.3f sec\n",  (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) * 1e-9);

    clock_gettime(CLOCK_MONOTONIC, &start);
    count_simd(counts);
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("SIMD time: %.3f sec\n",  (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) * 1e-9);

    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < NUM_THREADS; i++){
        thread_ids[i] = i;
        if (pthread_create(&threads[i], NULL, thread_simd_worker, &thread_ids[i]) != 0){
            printf("thread create failed\n");
            exit(1);
        }
    }
    for (int i = 0; i < NUM_THREADS; i++){
        if (pthread_join(threads[i], NULL) != 0){
            printf("thread join failed\n");
            exit(1);
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    for(int i = 0; i < 4; ++i)
        counts[i] = 0;
    for (int i = 0; i < NUM_THREADS; i++){
        counts[0] += partial_counts[i][0];
        counts[1] += partial_counts[i][1];
        counts[2] += partial_counts[i][2];
        counts[3] += partial_counts[i][3];
    }
    printf("SIMD + Multithreading time: %.3f sec\n",  (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) * 1e-9);

    pthread_mutex_destroy(&mutex);
    free(dna);

    return 0;
}
