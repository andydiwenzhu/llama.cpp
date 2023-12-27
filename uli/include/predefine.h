#include "llama.h"
#include <vector>

// params
typedef gpt_params ULI_PARAMS;

// token
typedef llama_token ULI_TOKEN;
typedef std::vector<llama_token> ULI_TOKENS;

// model
typedef llama_model* ULI_MODEL;

// kv cache
typedef void* ULI_KV;

// context
typedef struct ULI_CTX {
    llama_context* lctx;
    llama_sampling_context* sctx;
} ULI_CTX;

