#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <immintrin.h>

#define BUFFER_SIZE_MB 256
#define BUFFER_SIZE ((size_t)BUFFER_SIZE_MB * 1024 * 1024)
#define NUM_THREADS 4

typedef struct {
    char *buf;
    size_t start;
    size_t end;
} thread_arg;

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void generate_buffer(char *buf, size_t size)
{
    static const char charset[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()_+-=[]{}|;:,.<>? ";
    int len = sizeof(charset) - 1;
    srand(42);
    for (size_t i = 0; i < size; i++)
        buf[i] = charset[rand() % len];
}

static void *thread_func(void *arg)
{
    thread_arg *th = (thread_arg *)arg;
    for (size_t i = th->start; i < th->end; i++) {
        if (th->buf[i] >= 'a' && th->buf[i] <= 'z')
            th->buf[i] -= 32;
    }
    return NULL;
}

static void convert_simd(char *buf, size_t size)
{
    __m256i vec_a    = _mm256_set1_epi8('a');
    __m256i vec_z    = _mm256_set1_epi8('z');
    __m256i vec_diff = _mm256_set1_epi8(32);

    size_t i = 0;
    for (; i + 32 <= size; i += 32) {
        __m256i chunk = _mm256_loadu_si256((__m256i *)(buf + i));

        __m256i ge_a = _mm256_cmpgt_epi8(chunk, _mm256_sub_epi8(vec_a, _mm256_set1_epi8(1)));
        __m256i le_z = _mm256_cmpgt_epi8(_mm256_add_epi8(vec_z, _mm256_set1_epi8(1)), chunk);
        __m256i mask = _mm256_and_si256(ge_a, le_z);

        __m256i converted = _mm256_sub_epi8(chunk, _mm256_and_si256(mask, vec_diff));
        _mm256_storeu_si256((__m256i *)(buf + i), converted);
    }

    for (; i < size; i++) {
        if (buf[i] >= 'a' && buf[i] <= 'z')
            buf[i] -= 32;
    }
}

static void *thread_func_simd(void *arg)
{
    thread_arg *th = (thread_arg *)arg;
    convert_simd(th->buf + th->start, th->end - th->start);
    return NULL;
}

int main()
{
    char *buf_mt   = malloc(BUFFER_SIZE);
    char *buf_simd = malloc(BUFFER_SIZE);
    char *buf_simd_mt = malloc(BUFFER_SIZE);
    if(!buf_mt || !buf_simd || !buf_simd_mt){
        printf("malloc failed");
        return 1;
    }
    
    printf("Generating random buffer.\n");
    generate_buffer(buf_mt, BUFFER_SIZE);
    memcpy(buf_simd,    buf_mt, BUFFER_SIZE);
    memcpy(buf_simd_mt, buf_mt, BUFFER_SIZE);
    printf("Finished.\n");
    printf("Buffer size: %d MB\n", BUFFER_SIZE_MB);
    printf("Threads used: %d\n", NUM_THREADS);

    pthread_t threads[NUM_THREADS];
    thread_arg args[NUM_THREADS];
    size_t chunk = BUFFER_SIZE / NUM_THREADS;

    double start_sec = now_sec();
    for(int i = 0; i < NUM_THREADS; i++){
        args[i].buf   = buf_mt;
        args[i].start = i * chunk;
        args[i].end   = (i == NUM_THREADS - 1) ? BUFFER_SIZE : (i + 1) * chunk;
        if(pthread_create(&threads[i], NULL, thread_func, &args[i]) != 0){
            printf("thread create failed\n");
            exit(1);
        }
    }
    for(int i = 0; i < NUM_THREADS; i++){
        if(pthread_join(threads[i], NULL) != 0){
            printf("thread join failed\n");
            exit(1);
        }
    }
    double time_diff = now_sec() - start_sec;
    printf("Multithreading time: %.3f sec\n", time_diff);

    start_sec = now_sec();
    convert_simd(buf_simd, BUFFER_SIZE);
    time_diff = now_sec() - start_sec;
    printf("SIMD time: %.3f sec\n", time_diff);

    start_sec = now_sec();
    for (int i = 0; i < NUM_THREADS; i++) {
        args[i].buf   = buf_simd_mt;
        args[i].start = i * chunk;
        args[i].end   = (i == NUM_THREADS - 1) ? BUFFER_SIZE : (i + 1) * chunk;
        if (pthread_create(&threads[i], NULL, thread_func_simd, &args[i]) != 0) {
            printf("thread create failed\n");
            exit(1);
        }
    }
    for (int i = 0; i < NUM_THREADS; i++) {
        if (pthread_join(threads[i], NULL) != 0) {
            printf("thread join failed\n");
            exit(1);
        }
    }
    double simd_mt_time = now_sec() - start_sec;
    printf("SIMD + Multithreading: %.3f sec\n", simd_mt_time);

    free(buf_mt);
    free(buf_simd);
    free(buf_simd_mt);
    return 0;
}
