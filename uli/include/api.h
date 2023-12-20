#include "predefine.h"

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

// token
const char* uli_tokens_to_string(ULI_TOKENS tokens);

ULI_TOKENS uli_string_to_tokens(const char* s);

// model
ULI_MODEL uli_init_model(const char* model_path);

ULI_MODEL_PARAMS uli_get_model_params(ULI_MODEL model);

// kv

// context
ULI_CTX uli_init_context(ULI_MODEL model);



ULI_TOKEN uli_prefill(ULI_CTX ctx, ULI_TOKENS tokens);

ULI_TOKEN uli_decode(ULI_CTX ctx, ULI_TOKEN token);

void uli_clear(ULI_CTX ctx);
