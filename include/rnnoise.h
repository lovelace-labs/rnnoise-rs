/* rnnoise-rs C ABI — compatible subset of upstream rnnoise.h.
 * Build the library with `cargo build --release --features capi`.
 *
 * Use the create/process/destroy flow; rnnoise_get_size/rnnoise_init from the
 * C library are intentionally not provided (the Rust state is opaque). */

#ifndef RNNOISE_RS_H
#define RNNOISE_RS_H 1

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct DenoiseState DenoiseState;
typedef struct RNNModel RNNModel;

/** Number of samples processed by rnnoise_process_frame at a time (480). */
int rnnoise_get_frame_size(void);

/** Allocate and initialise a DenoiseState. If model is NULL the default model
 *  is used. Free with rnnoise_destroy(). */
DenoiseState *rnnoise_create(RNNModel *model);

/** Free a DenoiseState produced by rnnoise_create. */
void rnnoise_destroy(DenoiseState *st);

/** Denoise a frame of samples. `out` and `in` must each be at least
 *  rnnoise_get_frame_size() floats. Returns the VAD probability. */
float rnnoise_process_frame(DenoiseState *st, float *out, const float *in);

/** Load a model from a memory buffer (must outlive use). NULL on failure.
 *  Free with rnnoise_model_free(). */
RNNModel *rnnoise_model_from_buffer(const void *ptr, int len);

/** Load a model from a file name. NULL on failure. Free with
 *  rnnoise_model_free(). */
RNNModel *rnnoise_model_from_filename(const char *filename);

/** Free a model handle from rnnoise_model_from_*. */
void rnnoise_model_free(RNNModel *model);

#ifdef __cplusplus
}
#endif

#endif /* RNNOISE_RS_H */
