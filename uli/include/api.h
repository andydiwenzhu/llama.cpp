// token
const char* uli_tokens_to_string(ULI_TOKENS tokens);

ULI_TOKENS uli_string_to_tokens(const char* s);


// model
ULI_MODEL uli_init_model(const char* model_path);

ULI_MODEL_PARAMS uli_get_model_params(ULI_MODEL model);


// kv
ULI_KV uli_set_k(ULI_KV kv, size_t idx_token, size_t n_token, int8_t* data, size_t size);

ULI_KV uli_set_v(ULI_KV kv, size_t idx_token, size_t n_token, int8_t* data, size_t size);

int8_t* uli_get_k(ULI_KV kv, size_t idx_layer, size_t* size);

int8_t* uli_get_v(ULI_KV kv, size_t idx_layer, size_t* size);


// context
ULI_CTX uli_init_context(ULI_MODEL model);

ULI_KV uli_get_kv(ULI_CTX ctx, size_t idx_layer);

void ulit_set_kv(ULI_CTX ctx, size_t idx_layer, ULI_KV kv);

ULI_TOKEN uli_prefill(ULI_CTX ctx, ULI_TOKENS tokens);

ULI_TOKEN uli_decode(ULI_CTX ctx, ULI_TOKEN token);

void uli_clear(ULI_CTX ctx);


