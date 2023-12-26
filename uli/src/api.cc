#include "predefine.h"
#include "api.h"
#include "llama.h"


// token
const char* uli_tokens_to_string(ULI_TOKENS tokens);

ULI_TOKENS uli_string_to_tokens(const char* s);


// model
ULI_MODEL uli_init_model(const char* model_path) {
    auto mparams = llama_model_default_params();
    llama_model * model  = llama_load_model_from_file(model_path, mparams);
    return model;
}

ULI_MODEL_PARAMS uli_get_model_params(ULI_MODEL model) {
    auto _model = static_cast<llama_model*>(model);
    ULI_MODEL_PARAMS params;
    params.n_ctx_train = 1;
    params.n_embd = 2;
    params.n_gqa = 3;
    params.n_layer = 4;
    params.w_sz = 1;
    return params;
}


// kv
ULI_KV uli_set_k(ULI_KV kv, size_t idx_token, size_t n_token, int8_t* data, size_t size);

ULI_KV uli_set_v(ULI_KV kv, size_t idx_token, size_t n_token, int8_t* data, size_t size);

int8_t* uli_get_k(ULI_KV kv, size_t idx_layer, size_t* size);

int8_t* uli_get_v(ULI_KV kv, size_t idx_layer, size_t* size);


// context
ULI_CTX uli_init_context(ULI_MODEL model) {
    auto _model = static_cast<llama_model*>(model);
    auto cparams = llama_context_default_params();

    cparams.n_threads = 64;
    llama_context * lctx = llama_new_context_with_model(_model, cparams);
    return lctx;
}

ULI_KV uli_get_kv(ULI_CTX ctx, size_t idx_layer) {

}

void uli_set_kv(ULI_CTX ctx, size_t idx_layer, ULI_KV kv) {

}

size_t uli_get_kv_n_token(ULI_CTX ctx) {

}

ULI_TOKEN uli_prefill(ULI_CTX ctx, ULI_TOKENS tokens) {

}

ULI_TOKEN uli_decode(ULI_CTX ctx, ULI_TOKEN token) {

}

void uli_clear(ULI_CTX ctx) {

}
