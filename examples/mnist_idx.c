#include "tkm.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    float *images;
    uint8_t *labels;
    uint32_t count;
} MnistData;

static uint32_t read_be32(const uint8_t bytes[4]) {
    return ((uint32_t)bytes[0] << 24) |
           ((uint32_t)bytes[1] << 16) |
           ((uint32_t)bytes[2] << 8) |
           (uint32_t)bytes[3];
}

static int read_exact(FILE *file, void *buffer, size_t size) {
    return fread(buffer, 1, size, file) == size;
}

static int load_mnist_idx(const char *image_path, const char *label_path, MnistData *data) {
    FILE *images_file = NULL;
    FILE *labels_file = NULL;
    uint8_t image_header[16], label_header[8];
    uint8_t *raw_images = NULL;
    size_t pixel_count, image_bytes;
    int ok = 0;

    images_file = fopen(image_path, "rb");
    if (!images_file) {
        perror(image_path);
        goto cleanup;
    }
    labels_file = fopen(label_path, "rb");
    if (!labels_file) {
        perror(label_path);
        goto cleanup;
    }

    if (!read_exact(images_file, image_header, sizeof(image_header)) ||
        !read_exact(labels_file, label_header, sizeof(label_header))) {
        fprintf(stderr, "IDX header is truncated.\n");
        goto cleanup;
    }

    uint32_t image_count = read_be32(&image_header[4]);
    uint32_t rows = read_be32(&image_header[8]);
    uint32_t cols = read_be32(&image_header[12]);
    uint32_t label_count = read_be32(&label_header[4]);
    if (read_be32(image_header) != 2051 || read_be32(label_header) != 2049 ||
        image_count == 0 || image_count != label_count || rows != 28 || cols != 28) {
        fprintf(stderr, "Invalid MNIST IDX files (expected matching 28x28 images and labels).\n");
        goto cleanup;
    }

    pixel_count = (size_t)rows * cols;
    if ((size_t)image_count > SIZE_MAX / pixel_count ||
        (size_t)image_count * pixel_count > SIZE_MAX / sizeof(float)) {
        fprintf(stderr, "MNIST IDX dimensions exceed addressable memory.\n");
        goto cleanup;
    }
    image_bytes = (size_t)image_count * pixel_count;
    raw_images = (uint8_t*)malloc(image_bytes);
    data->images = (float*)malloc(image_bytes * sizeof(float));
    data->labels = (uint8_t*)malloc(image_count);
    if (!raw_images || !data->images || !data->labels) {
        fprintf(stderr, "Unable to allocate MNIST data buffers.\n");
        goto cleanup;
    }

    if (!read_exact(images_file, raw_images, image_bytes) ||
        !read_exact(labels_file, data->labels, image_count)) {
        fprintf(stderr, "IDX data is truncated.\n");
        goto cleanup;
    }
    for (size_t i = 0; i < image_bytes; i++) {
        data->images[i] = (float)raw_images[i] / 255.0f;
    }
    for (uint32_t i = 0; i < image_count; i++) {
        if (data->labels[i] > 9) {
            fprintf(stderr, "Invalid MNIST label %u at index %u.\n", data->labels[i], i);
            goto cleanup;
        }
    }
    data->count = image_count;
    ok = 1;

cleanup:
    free(raw_images);
    if (images_file) fclose(images_file);
    if (labels_file) fclose(labels_file);
    if (!ok) {
        free(data->images);
        free(data->labels);
        data->images = NULL;
        data->labels = NULL;
        data->count = 0;
    }
    return ok;
}

static void free_mnist_data(MnistData *data) {
    free(data->images);
    free(data->labels);
    data->images = NULL;
    data->labels = NULL;
    data->count = 0;
}

static int parse_positive_int(const char *text, int *value) {
    char *end = NULL;
    long parsed = strtol(text, &end, 10);
    if (!text[0] || !end || *end || parsed <= 0 || parsed > 1000000) return 0;
    *value = (int)parsed;
    return 1;
}

int main(int argc, char **argv) {
    if (argc < 5 || argc > 7) {
        fprintf(stderr, "Usage: %s <train-images-idx3> <train-labels-idx1> <test-images-idx3> <test-labels-idx1> [batch=64] [epochs=1]\n", argv[0]);
        return 2;
    }

    int batch_size = 64, epochs = 1;
    if ((argc >= 6 && !parse_positive_int(argv[5], &batch_size)) ||
        (argc >= 7 && !parse_positive_int(argv[6], &epochs))) {
        fprintf(stderr, "Batch size and epochs must be positive integers.\n");
        return 2;
    }

    MnistData train = {0}, test = {0};
    if (!load_mnist_idx(argv[1], argv[2], &train) ||
        !load_mnist_idx(argv[3], argv[4], &test)) {
        free_mnist_data(&train);
        free_mnist_data(&test);
        return 1;
    }

    TkmRuntime *runtime = tkm_load_file("models/resnet_mnist.tkm", batch_size);
    if (!runtime) {
        fprintf(stderr, "Could not load models/resnet_mnist.tkm.\n");
        free_mnist_data(&train);
        free_mnist_data(&test);
        return 1;
    }
    if (test.count > (uint32_t)INT_MAX) {
        fprintf(stderr, "Test set exceeds the runtime_eval count limit.\n");
        tkm_runtime_free(runtime);
        free_mnist_data(&train);
        free_mnist_data(&test);
        return 1;
    }

    uint32_t full_batches = train.count / (uint32_t)batch_size;
    if (full_batches == 0) {
        fprintf(stderr, "Training set must contain at least one full batch (%d samples).\n", batch_size);
        tkm_runtime_free(runtime);
        free_mnist_data(&train);
        free_mnist_data(&test);
        return 1;
    }
    if (train.count % (uint32_t)batch_size != 0) {
        fprintf(stderr, "Note: skipping %u training samples to keep each fixed-size training batch full.\n",
                train.count % (uint32_t)batch_size);
    }

    for (int epoch = 0; epoch < epochs; epoch++) {
        float loss_sum = 0.0f;
        for (uint32_t b = 0; b < full_batches; b++) {
            size_t start = (size_t)b * batch_size;
            const float *images = train.images + start * 28 * 28;
            const uint8_t *labels = train.labels + start;
            loss_sum += tkm_runtime_step(runtime, images, labels, 1e-3f);
        }
        printf("epoch %d/%d: train loss %.4f\n", epoch + 1, epochs, loss_sum / full_batches);
    }

    int correct = tkm_runtime_eval(runtime, test.images, test.labels, (int)test.count);
    printf("test accuracy: %d/%u (%.2f%%)\n", correct, test.count,
           100.0 * (double)correct / test.count);

    tkm_runtime_free(runtime);
    free_mnist_data(&train);
    free_mnist_data(&test);
    return 0;
}
