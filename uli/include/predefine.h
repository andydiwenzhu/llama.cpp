// Unified Llama Interface

#include <stddef.h>
#include <stdint.h>


// token
typedef void* ULI_TOKEN;
typedef struct ULI_TOKENS {
    ULI_TOKEN* tokens;
    size_t size;
} ULI_TOKENS;


// model
typedef void* ULI_MODEL;

typedef struct ULI_MODEL_PARAMS {
    size_t n_layer;
    size_t n_embd;
    size_t n_gqa;
    size_t w_sz;
    size_t n_ctx_train;
} ULI_MODEL_PARAMS;


// kv cache
typedef void* ULI_KV;


// context
typedef void* ULI_CTX;


