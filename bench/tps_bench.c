#define TESTING
#include "run.c"

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <checkpoint> [steps]\n", argv[0]);
        return 1;
    }
    char *checkpoint = argv[1];
    int steps = 128;
    if (argc >= 3) {
        steps = atoi(argv[2]);
        if (steps <= 0) {
            fprintf(stderr, "steps must be positive\n");
            return 1;
        }
    }

    Transformer transformer = {0};
    build_transformer(&transformer, checkpoint);

    int max_steps = transformer.config.seq_len;
    if (steps > max_steps) {
        steps = max_steps;
    }

    int token = 0;
    int pos = 0;

    long start = time_in_ms();
    while (pos < steps) {
        forward(&transformer, token % transformer.config.vocab_size, pos);
        pos++;
        token++;
    }
    long end = time_in_ms();

    double elapsed_ms = (double)(end - start);
    double tps = elapsed_ms > 0 ? (double)steps / (elapsed_ms / 1000.0) : 0.0;
    printf("achieved tok/s: %.6f\n", tps);

    free_transformer(&transformer);
    return 0;
}
