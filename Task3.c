#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <pthread.h>
#include <immintrin.h>
#include <fcntl.h>
#include <unistd.h>

#define NUM_THREADS 4

typedef struct {
    const uint8_t *source;
    uint8_t *destination;
    size_t start;
    size_t end;
} thread_arg;

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static uint8_t *read_ppm(const char *path, int *width, int *height)
{
    int fd = open(path, O_RDONLY);
    if (fd == -1) {
        perror("open failed");
        exit(1);
    }

    char header[64];
    ssize_t n = read(fd, header, sizeof(header) - 1);
    if (n <= 0) {
        perror("read failed");
        exit(1);
    }
    header[n] = '\0';

    int maxval, header_len;
    sscanf(header, "P6 %d %d %d%n", width, height, &maxval, &header_len);
    header_len++;

    size_t size = (size_t)(*width) * (*height) * 3;
    uint8_t *data = malloc(size);
    if (!data) {
        perror("malloc failed");
        exit(1);
    }

    if (lseek(fd, header_len, SEEK_SET) == -1) {
        perror("lseek failed");
        exit(1);
    }

    size_t total = 0;
    while (total < size) {
        ssize_t r = read(fd, data + total, size - total);
        if (r <= 0) {
            perror("read pixels");
            exit(1);
        }
        total += r;
    }

    if (close(fd) == -1)
        perror("close");

    return data;
}

static void write_ppm(const char *path, uint8_t *data, int width, int height)
{
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd == -1) {
        perror("open output");
        exit(1);
    }

    char header[64];
    int hlen = snprintf(header, sizeof(header), "P6\n%d %d\n255\n", width, height);
    if (write(fd, header, hlen) == -1) {
        perror("write header");
        exit(1);
    }

    size_t size = (size_t)width * height * 3;
    size_t total = 0;
    while (total < size) {
        ssize_t w = write(fd, data + total, size - total);
        if (w == -1) {
            perror("write pixels");
            exit(1);
        }
        total += w;
    }

    if (close(fd) == -1)
        perror("close");
}

static void grayscale_scalar(const uint8_t *src, uint8_t *dst, size_t npixels)
{
    for (size_t i = 0; i < npixels; i++) {
        uint8_t r = src[i*3 + 0];
        uint8_t g = src[i*3 + 1];
        uint8_t b = src[i*3 + 2];
        uint8_t gray = (uint8_t)(0.299f * r + 0.587f * g + 0.114f * b);
        dst[i*3 + 0] = gray;
        dst[i*3 + 1] = gray;
        dst[i*3 + 2] = gray;
    }
}

static void grayscale_simd(const uint8_t *src, uint8_t *dst, size_t npixels)
{
    __m256 vec_wr = _mm256_set1_ps(0.299f);
    __m256 vec_wg = _mm256_set1_ps(0.587f);
    __m256 vec_wb = _mm256_set1_ps(0.114f);

    size_t i = 0;
    for (; i + 8 <= npixels; i += 8) {
        float r[8], g[8], b[8];
        for (int j = 0; j < 8; j++) {
            r[j] = src[(i+j)*3 + 0];
            g[j] = src[(i+j)*3 + 1];
            b[j] = src[(i+j)*3 + 2];
        }

        __m256 vr = _mm256_loadu_ps(r);
        __m256 vg = _mm256_loadu_ps(g);
        __m256 vb = _mm256_loadu_ps(b);

        __m256 vgray = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(vr, vec_wr), _mm256_mul_ps(vg, vec_wg)), _mm256_mul_ps(vb, vec_wb));

        vgray = _mm256_floor_ps(vgray);

        int tmp[8];
        _mm256_storeu_si256((__m256i *)tmp, _mm256_cvtps_epi32(vgray));

        for (int j = 0; j < 8; j++) {
            uint8_t gray = (uint8_t)tmp[j];
            dst[(i+j)*3 + 0] = gray;
            dst[(i+j)*3 + 1] = gray;
            dst[(i+j)*3 + 2] = gray;
        }
    }

    for (; i < npixels; i++) {
        uint8_t gray = (uint8_t)(0.299f*src[i*3] + 0.587f*src[i*3+1] + 0.114f*src[i*3+2]);
        dst[i*3+0] = dst[i*3+1] = dst[i*3+2] = gray;
    }
}

static void *thread_func(void *arg)
{
    thread_arg *th = (thread_arg *)arg;
    grayscale_scalar(th->source + th->start * 3,
                     th->destination + th->start * 3,
                     th->end - th->start);
    return NULL;
}

static void *thread_func_simd(void *arg)
{
    thread_arg *th = (thread_arg *)arg;
    grayscale_simd(th->source + th->start * 3,
                   th->destination + th->start * 3,
                   th->end - th->start);
    return NULL;
}

int main(int argc, char *argv[])
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s input.ppm\n", argv[0]);
        return 1;
    }

    int width, height;
    uint8_t *src = read_ppm(argv[1], &width, &height);
    size_t npixels = (size_t)width * height;
    size_t nbytes  = npixels * 3;

    uint8_t *out_scalar  = malloc(nbytes);
    uint8_t *out_simd    = malloc(nbytes);
    uint8_t *out_mt      = malloc(nbytes);
    uint8_t *out_mt_simd = malloc(nbytes);

    if (!out_scalar || !out_simd || !out_mt || !out_mt_simd) {
        perror("malloc failed");
        return 1;
    }

    printf("Image size: %d x %d\n", width, height);
    printf("Threads used: %d\n", NUM_THREADS);

    double start_time = now_sec();
    grayscale_scalar(src, out_scalar, npixels);
    printf("Scalar time: %.3f sec\n", now_sec() - start_time);

    start_time = now_sec();
    grayscale_simd(src, out_simd, npixels);
    printf("SIMD time: %.3f sec\n", now_sec() - start_time);

    pthread_t threads[NUM_THREADS];
    thread_arg args[NUM_THREADS];
    size_t chunk = npixels / NUM_THREADS;

    start_time = now_sec();
    for (int i = 0; i < NUM_THREADS; i++) {
        args[i].source = src;
        args[i].destination = out_mt;
        args[i].start = i * chunk;
        args[i].end = (i == NUM_THREADS - 1) ? npixels : (i + 1) * chunk;
        if (pthread_create(&threads[i], NULL, thread_func, &args[i]) != 0) {
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
    printf("Multithreading time: %.3f sec\n", now_sec() - start_time);

    start_time = now_sec();
    for (int i = 0; i < NUM_THREADS; i++) {
        args[i].source = src;
        args[i].destination = out_mt_simd;
        args[i].start = i * chunk;
        args[i].end = (i == NUM_THREADS - 1) ? npixels : (i + 1) * chunk;
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
    printf("Multithreading + SIMD time: %.3f sec\n", now_sec() - start_time);

    int verify = (memcmp(out_scalar, out_simd,    nbytes) == 0) && 
                 (memcmp(out_scalar, out_mt,      nbytes) == 0) &&
                 (memcmp(out_scalar, out_mt_simd, nbytes) == 0);
    printf("Verification: %s\n", verify ? "PASSED" : "FAILED");

    write_ppm("gray_output.ppm", out_scalar, width, height);
    printf("Output image: gray_output.ppm\n");

    free(src);
    free(out_scalar);
    free(out_simd);
    free(out_mt);
    free(out_mt_simd);
    return 0;
}
