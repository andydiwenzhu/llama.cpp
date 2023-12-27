#include "predefine.h"
#include "api.h"
#include "common.h"
#include "llama.h"

#include <string>

// params
ULI_PARAMS uli_init_params(int argc, char** argv) {
    ULI_PARAMS params;
    if (!gpt_params_parse(argc, argv, params)) {
        fprintf(stderr, "%s: error: failed to load params\n", __func__);
    }
    return params;
}

// token
std::string uli_token_to_string(ULI_CTX ctx, ULI_TOKEN token) {
    return llama_token_to_piece(ctx.lctx, token);
}

ULI_TOKENS uli_string_to_tokens(ULI_CTX ctx, const char* s) {
    return llama_tokenize(ctx.lctx, std::string(s), true, false);
}


// model
ULI_MODEL uli_init_model(ULI_PARAMS params) {
    auto mparams = llama_model_params_from_gpt_params(params);

    llama_model * model  = llama_load_model_from_file(params.model.c_str(), mparams);
    if (model == NULL) {
        fprintf(stderr, "%s: error: failed to load model '%s'\n", __func__, params.model.c_str());
    }

    return model;
}


// kv
ULI_KV uli_set_k(ULI_KV kv, size_t idx_token, size_t n_token, int8_t* data, size_t size);

ULI_KV uli_set_v(ULI_KV kv, size_t idx_token, size_t n_token, int8_t* data, size_t size);

int8_t* uli_get_k(ULI_KV kv, size_t idx_layer, size_t* size);

int8_t* uli_get_v(ULI_KV kv, size_t idx_layer, size_t* size);


// context
ULI_CTX uli_init_context(ULI_MODEL model, ULI_PARAMS params) {
    auto cparams = llama_context_params_from_gpt_params(params);

    llama_context * lctx = llama_new_context_with_model(model, cparams);
    if (lctx == NULL) {
        fprintf(stderr, "%s: error: failed to create context with model '%s'\n", __func__, params.model.c_str());
    }

    llama_sampling_context * sctx = llama_sampling_init(params.sparams);
    if (sctx == NULL) {
        fprintf(stderr, "%s: error: failed to create sampling context with model '%s'\n", __func__, params.model.c_str());
    }

    return ULI_CTX{lctx, sctx};
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
