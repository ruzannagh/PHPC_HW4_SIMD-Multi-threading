#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>
#include <immintrin.h>
#include <string.h>

#define BUF_SIZE (256 * 1024 * 1024)
#define NUM_THREADS 4

char *buf1;
char *buf2;
char *buf3;

void generate_buffer(char *buf) 
{
    for (int i = 0; i < BUF_SIZE; i++) {
        int r = rand() % 4;
        if (r == 0)      
		buf[i] = 'a' + rand() % 26;
        else if (r == 1) 
		buf[i] = 'A' + rand() % 26;
        else if (r == 2) 
		buf[i] = '0' + rand() % 10;
        else             
		buf[i] = "!@#$%^&*() "[rand() % 11];
    }
}

void *thread_worker(void *arg) 
{
    int id = *((int *)arg);
    int start = id * (BUF_SIZE / NUM_THREADS);
    int end = (id == NUM_THREADS - 1) ? BUF_SIZE : (id + 1) * (BUF_SIZE / NUM_THREADS);

    for (int i = start; i < end; i++){
        if (buf1[i] >= 'a' && buf1[i] <= 'z')
            buf1[i] -= 32;
    }
    return NULL;
}

void to_upper_simd(char *buf) 
{
    __m256i va  = _mm256_set1_epi8('a');
    __m256i vz  = _mm256_set1_epi8('z');
    __m256i v32 = _mm256_set1_epi8(32);

    for (int i = 0; i + 32 <= BUF_SIZE; i += 32){
        __m256i chunk = _mm256_loadu_si256((__m256i *)(buf + i));
        __m256i ge_a = _mm256_cmpgt_epi8(chunk, _mm256_sub_epi8(va, _mm256_set1_epi8(1)));
        __m256i le_z = _mm256_cmpgt_epi8(_mm256_add_epi8(vz, _mm256_set1_epi8(1)), chunk);
        __m256i mask = _mm256_and_si256(ge_a, le_z);
        __m256i result = _mm256_sub_epi8(chunk, _mm256_and_si256(mask, v32));
        _mm256_storeu_si256((__m256i *)(buf + i), result);
    }

    for (int i = 0; i < BUF_SIZE; i++){
        if (buf[i] >= 'a' && buf[i] <= 'z')
            buf[i] -= 32;
    }
}

void *thread_simd_worker(void *arg) 
{
    int id = *((int *)arg);
    int start = id * (BUF_SIZE / NUM_THREADS);
    int end = (id == NUM_THREADS - 1) ? BUF_SIZE : (id + 1) * (BUF_SIZE / NUM_THREADS);

    __m256i va = _mm256_set1_epi8('a');
    __m256i vz = _mm256_set1_epi8('z');
    __m256i v32 = _mm256_set1_epi8(32);

    for (int i = start; i + 32 <= end; i += 32) {
        __m256i chunk = _mm256_loadu_si256((__m256i *)(buf3 + i));
        __m256i ge_a = _mm256_cmpgt_epi8(chunk, _mm256_sub_epi8(va, _mm256_set1_epi8(1)));
        __m256i le_z = _mm256_cmpgt_epi8(_mm256_add_epi8(vz, _mm256_set1_epi8(1)), chunk);
        __m256i mask = _mm256_and_si256(ge_a, le_z);
        __m256i result = _mm256_sub_epi8(chunk, _mm256_and_si256(mask, v32));
        _mm256_storeu_si256((__m256i *)(buf3 + i), result);
    }

    for (int i = start; i < end; i++){
        if (buf3[i] >= 'a' && buf3[i] <= 'z')
            buf3[i] -= 32;
    }

    return NULL;
}

int main() 
{
    srand(42);

    buf1 = (char *)malloc(BUF_SIZE);
    buf2 = (char *)malloc(BUF_SIZE);
    buf3 = (char *)malloc(BUF_SIZE);

    if (!buf1 || !buf2 || !buf3) {
        printf("malloc failed");
        return 1;
    }

    generate_buffer(buf1);
    memcpy(buf2, buf1, BUF_SIZE);
    memcpy(buf3, buf1, BUF_SIZE);

    printf("Buffer size: %d MB\n", BUF_SIZE / 1024 / 1024);
    printf("Threads used: %d\n", NUM_THREADS);

    struct timespec start, end;
    pthread_t threads[NUM_THREADS];
    int thread_ids[NUM_THREADS];

    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < NUM_THREADS; i++){
        thread_ids[i] = i;
        if (pthread_create(&threads[i], NULL, thread_worker, &thread_ids[i]) != 0){
            printf("thread create failed\n");
            return 1;
        }
    }
    for (int i = 0; i < NUM_THREADS; i++){
        if (pthread_join(threads[i], NULL) != 0){
            printf("thread join failed\n");
            return 1;
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("Multithreading time: %.3f sec\n",
           (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) * 1e-9);

    clock_gettime(CLOCK_MONOTONIC, &start);
    to_upper_simd(buf2);
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("SIMD time: %.3f sec\n",
           (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) * 1e-9);

    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < NUM_THREADS; i++){
        thread_ids[i] = i;
        if (pthread_create(&threads[i], NULL, thread_simd_worker, &thread_ids[i]) != 0){
            printf("thread create failed\n");
            return 1;
        }
    }
    for (int i = 0; i < NUM_THREADS; i++){
        if (pthread_join(threads[i], NULL) != 0){
            printf("thread join failed\n");
            return 1;
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("SIMD + Multithreading: %.3f sec\n",
           (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) * 1e-9);

    free(buf1); 
    free(buf2); 
    free(buf3);

    return 0;
}
