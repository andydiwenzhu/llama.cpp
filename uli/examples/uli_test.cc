#include <assert.h>
#include <stdio.h>

#include "predefine.h"
#include "api.h"

int main(int argc, const char** argv) {
    assert(argc == 2);
    ULI_MODEL m = uli_init_model(argv[1]);
    ULI_CTX ctx = uli_init_context(m);
    return 0;
}