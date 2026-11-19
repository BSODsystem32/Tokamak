#include "tkm.h"

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

enum {
    CIFAR_IMAGE_BYTES = 3072,
    CIFAR_RECORD_BYTES = 3073,
    CIFAR_TRAIN_BATCHES = 5,
    CIFAR_TRAIN_IMAGES = 50000,
    CIFAR_TEST_IMAGES = 10000
};

typedef struct {
    float *images;
    uint8_t *labels;
    size_t count;
} CifarData;

static void free_cifar_data(CifarData *data);

static int read_exact(FILE *file, void *buffer, size_t size) {
    return fread(buffer, 1, size, file) == size;
}

static int load_cifar_file(const char *path, CifarData *data, size_t offset, size_t count) {
    FILE *file = fopen(path, "rb");
    uint8_t record[CIFAR_RECORD_BYTES];
    if (!file) {
        perror(path);
        return 0;
    }

    for (size_t image = 0; image < count; image++) {
        if (!read_exact(file, record, sizeof(record)) || record[0] > 9) {
            fprintf(stderr, "Invalid or truncated CIFAR-10 record in %s at image %zu.\n", path, image);
            fclose(file);
            return 0;
        }
        data->labels[offset + image] = record[0];
        for (int y = 0; y < 32; y++) {
            for (int x = 0; x < 32; x++) {
                size_t pixel = (size_t)y * 32 + x;
                size_t dst = pixel * 3;
                data->images[(offset + image) * CIFAR_IMAGE_BYTES + dst + 0] = record[1 + pixel] / 255.0f;
                data->images[(offset + image) * CIFAR_IMAGE_BYTES + dst + 1] = record[1 + 1024 + pixel] / 255.0f;
                data->images[(offset + image) * CIFAR_IMAGE_BYTES + dst + 2] = record[1 + 2048 + pixel] / 255.0f;
            }
        }
    }
    fclose(file);
    return 1;
}

static int load_cifar10(const char *directory, CifarData *train, CifarData *test) {
    char path[1024];
    memset(train, 0, sizeof(*train));
    memset(test, 0, sizeof(*test));
    train->images = (float*)malloc((size_t)CIFAR_TRAIN_IMAGES * CIFAR_IMAGE_BYTES * sizeof(float));
    train->labels = (uint8_t*)malloc(CIFAR_TRAIN_IMAGES);
    test->images = (float*)malloc((size_t)CIFAR_TEST_IMAGES * CIFAR_IMAGE_BYTES * sizeof(float));
    test->labels = (uint8_t*)malloc(CIFAR_TEST_IMAGES);
    if (!train->images || !train->labels || !test->images || !test->labels) {
        fprintf(stderr, "Unable to allocate CIFAR-10 buffers.\n");
        free_cifar_data(train);
        free_cifar_data(test);
        return 0;
    }
    for (int batch = 1; batch <= CIFAR_TRAIN_BATCHES; batch++) {
        snprintf(path, sizeof(path), "%s/data_batch_%d.bin", directory, batch);
        if (!load_cifar_file(path, train, (size_t)(batch - 1) * (CIFAR_TRAIN_IMAGES / CIFAR_TRAIN_BATCHES),
                             CIFAR_TRAIN_IMAGES / CIFAR_TRAIN_BATCHES)) {
            free_cifar_data(train);
            free_cifar_data(test);
            return 0;
        }
    }
    snprintf(path, sizeof(path), "%s/test_batch.bin", directory);
    if (!load_cifar_file(path, test, 0, CIFAR_TEST_IMAGES)) {
        free_cifar_data(train);
        free_cifar_data(test);
        return 0;
    }
    train->count = CIFAR_TRAIN_IMAGES;
    test->count = CIFAR_TEST_IMAGES;
    return 1;
}

static void free_cifar_data(CifarData *data) {
    free(data->images);
    free(data->labels);
    memset(data, 0, sizeof(*data));
}

static int parse_positive_int(const char *text, int *value) {
    char *end = NULL;
    long parsed = strtol(text, &end, 10);
    if (!text[0] || !end || *end || parsed <= 0 || parsed > 1000000) return 0;
    *value = (int)parsed;
    return 1;
}

typedef struct {
    uint64_t state;
    uint64_t inc;
} Pcg32;

static uint32_t pcg32_next(Pcg32 *rng) {
    uint64_t oldstate = rng->state;
    rng->state = oldstate * 6364136223846793005ULL + rng->inc;
    uint32_t xorshifted = (uint32_t)(((oldstate >> 18u) ^ oldstate) >> 27u);
    uint32_t rot = (uint32_t)(oldstate >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((32u - rot) & 31u));
}

static void pcg32_init(Pcg32 *rng, uint64_t initstate, uint64_t initseq) {
    rng->state = 0ULL;
    rng->inc = (initseq << 1u) | 1ULL;
    pcg32_next(rng);
    rng->state += initstate;
    pcg32_next(rng);
}

static uint32_t pcg32_bounded(Pcg32 *rng, uint32_t bound) {
    uint32_t threshold = (~bound + 1u) % bound;
    for (;;) {
        uint32_t r = pcg32_next(rng);
        if (r >= threshold) return r % bound;
    }
}

static int parse_positive_float(const char *text, float *value) {
    char *end = NULL;
    float parsed = strtof(text, &end);
    if (!text[0] || !end || *end || !isfinite(parsed) || parsed <= 0.0f) return 0;
    *value = parsed;
    return 1;
}

int main(int argc, char **argv) {
    if (argc < 2 || argc > 5) {
        fprintf(stderr, "Usage: %s <cifar-10-binary-dir> [batch=64] [epochs=1] [lr=0.02]\n", argv[0]);
        return 2;
    }
    int batch_size = 64, epochs = 1;
    float peak_lr = 0.02f;
    if ((argc >= 3 && !parse_positive_int(argv[2], &batch_size)) ||
        (argc >= 4 && !parse_positive_int(argv[3], &epochs)) ||
        (argc >= 5 && !parse_positive_float(argv[4], &peak_lr))) {
        fprintf(stderr, "Batch size and epochs must be positive integers; learning rate must be finite and positive.\n");
        return 2;
    }

    CifarData train = {0}, test = {0};
    if (!load_cifar10(argv[1], &train, &test)) {
        free_cifar_data(&train);
        free_cifar_data(&test);
        return 1;
    }
    TkmRuntime *runtime = tkm_load_file("models/resnet_cifar10.tkm", batch_size);
    if (!runtime) {
        fprintf(stderr, "Could not load models/resnet_cifar10.tkm.\n");
        free_cifar_data(&train);
        free_cifar_data(&test);
        return 1;
    }

    size_t full_batches = train.count / (size_t)batch_size;
    if (full_batches == 0) {
        fprintf(stderr, "Training set must contain at least one full batch.\n");
        tkm_runtime_free(runtime);
        free_cifar_data(&train);
        free_cifar_data(&test);
        return 1;
    }
    size_t total_steps = full_batches * (size_t)epochs;
    size_t warmup_steps = total_steps / 10;
    if (warmup_steps > 500) warmup_steps = 500;

    size_t *order = (size_t*)malloc(train.count * sizeof(*order));
    float *batch_images = (float*)malloc((size_t)batch_size * CIFAR_IMAGE_BYTES * sizeof(*batch_images));
    uint8_t *batch_labels = (uint8_t*)malloc((size_t)batch_size * sizeof(*batch_labels));
    if (!order || !batch_images || !batch_labels) {
        fprintf(stderr, "Unable to allocate CIFAR-10 shuffle buffers.\n");
        free(order);
        free(batch_images);
        free(batch_labels);
        tkm_runtime_free(runtime);
        free_cifar_data(&train);
        free_cifar_data(&test);
        return 1;
    }
    for (size_t i = 0; i < train.count; i++) order[i] = i;

    Pcg32 rng;
    pcg32_init(&rng, 0x4d595df4d0f33173ULL, 0x9e3779b97f4a7c15ULL);

    size_t step = 0;
    for (int epoch = 0; epoch < epochs; epoch++) {
        for (size_t i = train.count; i > 1; i--) {
            uint32_t j = pcg32_bounded(&rng, (uint32_t)i);
            size_t tmp = order[i - 1];
            order[i - 1] = order[j];
            order[j] = tmp;
        }
        float loss_sum = 0.0f;
        for (size_t b = 0; b < full_batches; b++, step++) {
            size_t start = b * (size_t)batch_size;
            for (size_t row = 0; row < (size_t)batch_size; row++) {
                size_t source = order[start + row];
                memcpy(batch_images + row * CIFAR_IMAGE_BYTES,
                       train.images + source * CIFAR_IMAGE_BYTES,
                       CIFAR_IMAGE_BYTES * sizeof(*batch_images));
                batch_labels[row] = train.labels[source];
            }
            float progress = total_steps > 1 ? (float)step / (float)(total_steps - 1) : 1.0f;
            float warmup = (warmup_steps > 0 && step < warmup_steps)
                ? (float)(step + 1) / (float)warmup_steps
                : 1.0f;
            float cosine = 0.5f * (1.0f + cosf(3.14159265359f * progress));
            float lr = peak_lr * warmup * (0.05f + 0.95f * cosine);
            loss_sum += tkm_runtime_step(runtime, batch_images, batch_labels, lr);
        }
        printf("epoch %d/%d: train loss %.4f\n", epoch + 1, epochs, loss_sum / (float)full_batches);
    }

    int correct = tkm_runtime_eval(runtime, test.images, test.labels, (int)test.count);
    printf("test accuracy: %d/%zu (%.2f%%)\n", correct, test.count,
           100.0 * (double)correct / (double)test.count);
    free(order);
    free(batch_images);
    free(batch_labels);
    tkm_runtime_free(runtime);
    free_cifar_data(&train);
    free_cifar_data(&test);
    return 0;
}
