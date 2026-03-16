#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <immintrin.h>
#include <math.h>

#define NUM_THREADS 4

typedef struct {
    int width;
    int height;
    unsigned char *data;
} Image;

Image input_img;
unsigned char *out_scalar;
unsigned char *out_simd;
unsigned char *out_mt;
unsigned char *out_mt_simd;

typedef struct {
    int id;
    unsigned char *in;
    unsigned char *out;
    long start_pixel;
    long end_pixel;
} ThreadArg;

Image read_ppm(const char *filename)
{
    Image img = {0, 0, NULL};
    FILE *f = fopen(filename, "rb");
    if(!f) { 
	printf("Cannot open %s\n", filename); 
	exit(1);
    }

    char magic[3];
    
    if(fscanf(f, "%2s", magic) != 1) {
        printf("fscanf failed\n");
        exit(1);
    }

    if(strcmp(magic, "P6") != 0){
	printf("Only P6 PPM supported\n"); 
	exit(1); 
    }

    int maxval;
    
    if(fscanf(f, "%d %d %d", &img.width, &img.height, &maxval) != 3){
        printf("fscanf failed\n");
        exit(1);
    }

    fgetc(f);

    long size = (long)img.width * img.height * 3;
    img.data = (unsigned char *)malloc(size);
    if(!img.data) { 
	printf("malloc failed\n"); 
	exit(1); 
    }
    if (fread(img.data, 1, size, f) != (size_t)size){
        printf("fread failed\n");
        exit(1);
    }

    fclose(f);
    return img;
}

void write_ppm(const char *filename, unsigned char *data, int width, int height)
{
    FILE *f = fopen(filename, "wb");
    if(!f){ 
	printf("open failed"); 
	exit(1);
    }

    fprintf(f, "P6\n%d %d\n255\n", width, height);
    long size = (long)width * height * 3;
    if(fwrite(data, 1, size, f) != (size_t)size) {
        printf("fwrite failed\n");
        exit(1);
    }

    fclose(f);
}

void generate_ppm(const char *filename, int width, int height)
{
    FILE *f = fopen(filename, "wb");
    if(!f) { 
	printf("Cannot create test image\n"); 
	exit(1); 
    }

    fprintf(f, "P6\n%d %d\n255\n", width, height);
    long pixels = (long)width * height;
    for(long i = 0; i < pixels; i++) {
        unsigned char r = rand() % 256;
        unsigned char g = rand() % 256;
        unsigned char b = rand() % 256;
        if (fwrite(&r, 1, 1, f) != 1 || fwrite(&g, 1, 1, f) != 1 || fwrite(&b, 1, 1, f) != 1) {
            printf("fwrite failed when generating an image\n");
            exit(1);
        }
    }

    fclose(f);
}

void grayscale_scalar(unsigned char *in, unsigned char *out, int width, int height)
{
    long pixels = (long)width * height;
    for(long i = 0; i < pixels; i++){
        unsigned char r = in[i * 3 + 0];
        unsigned char g = in[i * 3 + 1];
        unsigned char b = in[i * 3 + 2];
        unsigned char gray = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
        out[i * 3 + 0] = gray;
        out[i * 3 + 1] = gray;
        out[i * 3 + 2] = gray;
    }
}

void grayscale_simd(unsigned char *in, unsigned char *out, int width, int height)
{
    long pixels = (long)width * height;

    __m256 vr_coeff = _mm256_set1_ps(0.299f);
    __m256 vg_coeff = _mm256_set1_ps(0.587f);
    __m256 vb_coeff = _mm256_set1_ps(0.114f);

    for (long i = 0; i + 8 <= pixels; i += 8) {
        float r[8], g[8], b[8];
        for (int j = 0; j < 8; j++) {
            r[j] = in[(i + j) * 3 + 0];
            g[j] = in[(i + j) * 3 + 1];
            b[j] = in[(i + j) * 3 + 2];
        }

        __m256 vr = _mm256_loadu_ps(r);
        __m256 vg = _mm256_loadu_ps(g);
        __m256 vb = _mm256_loadu_ps(b);

        __m256 vgray = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(vr, vr_coeff), _mm256_mul_ps(vg, vg_coeff)), _mm256_mul_ps(vb, vb_coeff));

        float gray[8];
        _mm256_storeu_ps(gray, vgray);

        for (int j = 0; j < 8; j++){
            unsigned char gv = (unsigned char)gray[j];
            out[(i + j) * 3 + 0] = gv;
            out[(i + j) * 3 + 1] = gv;
            out[(i + j) * 3 + 2] = gv;
        }
    }

    for (long i = 0; i < pixels; i++) {
        unsigned char r = in[i * 3 + 0];
        unsigned char g = in[i * 3 + 1];
        unsigned char b = in[i * 3 + 2];
        unsigned char gray = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
        out[i * 3 + 0] = gray;
        out[i * 3 + 1] = gray;
        out[i * 3 + 2] = gray;
    }
}

void *mt_worker(void *arg)
{
    ThreadArg *a = (ThreadArg *)arg;
    for(long i = a->start_pixel; i < a->end_pixel; i++){
        unsigned char r = a->in[i * 3 + 0];
        unsigned char g = a->in[i * 3 + 1];
        unsigned char b = a->in[i * 3 + 2];
        unsigned char gray = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
        a->out[i * 3 + 0] = gray;
        a->out[i * 3 + 1] = gray;
        a->out[i * 3 + 2] = gray;
    }

    return NULL;
}

void grayscale_mt(unsigned char *in, unsigned char *out, int width, int height)
{
    long pixels = (long)width * height;
    pthread_t threads[NUM_THREADS];
    ThreadArg args[NUM_THREADS];

    for(int i = 0; i < NUM_THREADS; i++){
        args[i].id = i;
        args[i].in = in;
        args[i].out = out;
        args[i].start_pixel = i * (pixels / NUM_THREADS);
        args[i].end_pixel = (i == NUM_THREADS - 1) ? pixels : (i + 1) * (pixels / NUM_THREADS);
        if(pthread_create(&threads[i], NULL, mt_worker, &args[i]) != 0) {
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
}

void *mt_simd_worker(void *arg)
{
    ThreadArg *a = (ThreadArg *)arg;

    __m256 vr_coeff = _mm256_set1_ps(0.299f);
    __m256 vg_coeff = _mm256_set1_ps(0.587f);
    __m256 vb_coeff = _mm256_set1_ps(0.114f);

    for(long i = a->start_pixel; i + 8 <= a->end_pixel; i += 8) {
        float r[8], g[8], b[8];
        for (int j = 0; j < 8; j++) {
            r[j] = a->in[(i + j) * 3 + 0];
            g[j] = a->in[(i + j) * 3 + 1];
            b[j] = a->in[(i + j) * 3 + 2];
        }

        __m256 vr = _mm256_loadu_ps(r);
        __m256 vg = _mm256_loadu_ps(g);
        __m256 vb = _mm256_loadu_ps(b);

        __m256 vgray = _mm256_add_ps(
            _mm256_add_ps(_mm256_mul_ps(vr, vr_coeff), _mm256_mul_ps(vg, vg_coeff)),
            _mm256_mul_ps(vb, vb_coeff)
        );

        float gray[8];
        _mm256_storeu_ps(gray, vgray);

        for(int j = 0; j < 8; j++) {
            unsigned char gv = (unsigned char)gray[j];
            a->out[(i + j) * 3 + 0] = gv;
            a->out[(i + j) * 3 + 1] = gv;
            a->out[(i + j) * 3 + 2] = gv;
        }
    }

    for(long i = a->start_pixel; i < a->end_pixel; i++) {
        unsigned char r = a->in[i * 3 + 0];
        unsigned char g = a->in[i * 3 + 1];
        unsigned char b = a->in[i * 3 + 2];
        unsigned char gray = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
        a->out[i * 3 + 0] = gray;
        a->out[i * 3 + 1] = gray;
        a->out[i * 3 + 2] = gray;
    }

    return NULL;
}

void grayscale_mt_simd(unsigned char *in, unsigned char *out, int width, int height)
{
    long pixels = (long)width * height;
    pthread_t threads[NUM_THREADS];
    ThreadArg args[NUM_THREADS];

    for(int i = 0; i < NUM_THREADS; i++){
        args[i].id = i;
        args[i].in = in;
        args[i].out = out;
        args[i].start_pixel = i * (pixels / NUM_THREADS);
        args[i].end_pixel = (i == NUM_THREADS - 1) ? pixels : (i + 1) * (pixels / NUM_THREADS);
        if(pthread_create(&threads[i], NULL, mt_simd_worker, &args[i]) != 0){
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
}

int verify(unsigned char *a, unsigned char *b, long size)
{
    for (long i = 0; i < size; i++) {
        if(abs((int)a[i] - (int)b[i]) > 1)
            return 0;
    }

    return 1;
}

int main(int argc, char *argv[])
{
    srand(42);

    const char *input_file = "input.ppm";

    if (argc < 2) {
        printf("No input file provided, generating a 3840x2160 test image...\n");
        generate_ppm(input_file, 3840, 2160);
    } 
    else {
        input_file = argv[1];
    }

    input_img = read_ppm(input_file);
    long size = (long)input_img.width * input_img.height * 3;

    printf("Image size: %d x %d\n", input_img.width, input_img.height);
    printf("Threads used: %d\n", NUM_THREADS);

    out_scalar = (unsigned char *)malloc(size);
    out_simd = (unsigned char *)malloc(size);
    out_mt = (unsigned char *)malloc(size);
    out_mt_simd = (unsigned char *)malloc(size);

    if (!out_scalar || !out_simd || !out_mt || !out_mt_simd) {
        printf("malloc failed\n");
        return 1;
    }

    struct timespec start, end;

    clock_gettime(CLOCK_MONOTONIC, &start);
    grayscale_scalar(input_img.data, out_scalar, input_img.width, input_img.height);
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("Scalar time: %.3f sec\n", (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) * 1e-9);

    clock_gettime(CLOCK_MONOTONIC, &start);
    grayscale_simd(input_img.data, out_simd, input_img.width, input_img.height);
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("SIMD time: %.3f sec\n", (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) * 1e-9);

    clock_gettime(CLOCK_MONOTONIC, &start);
    grayscale_mt(input_img.data, out_mt, input_img.width, input_img.height);
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("Multithreading time: %.3f sec\n", (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) * 1e-9);

    clock_gettime(CLOCK_MONOTONIC, &start);
    grayscale_mt_simd(input_img.data, out_mt_simd, input_img.width, input_img.height);
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("Multithreading + SIMD time: %.3f sec\n", (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) * 1e-9);

    int ok = verify(out_scalar, out_simd, size) &&
             verify(out_scalar, out_mt, size) &&
             verify(out_scalar, out_mt_simd, size);
    printf("Verification: %s\n", ok ? "PASSED" : "FAILED");

    write_ppm("gray_output.ppm", out_scalar, input_img.width, input_img.height);
    printf("Output image: gray_output.ppm\n");

    free(input_img.data);
    free(out_scalar);
    free(out_simd);
    free(out_mt);
    free(out_mt_simd);

    return 0;
}
