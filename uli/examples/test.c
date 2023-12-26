#include "predefine.h"
#include "api.h"

int main(int argc, const char** argv) {
    assert(argc == 2);
    ULI_MODEL m = uli_init_model(argv[1]);
    ULI_MODEL_PARAMS mp = uli_get_model_params(m);
    printf("[DEBUG] n_layer = %zu, n_ctx_train = %zu, n_embd = %zu, n_gqa = %zu, w_sz = %zu\n", mp.n_layer, mp.n_ctx_train, mp.n_embd, mp.n_gqa, mp.w_sz);
    ULI_CTX ctx = uli_init_context(m);
    return 0;
}