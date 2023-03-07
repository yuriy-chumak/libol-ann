#include "af.h"

// activation functions
// https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html
// #define tanh tanh_
// #define __builtin_tanh __builtin_tanh__
#include <math.h>
#undef tanh

#define ONE\
    __builtin_choose_expr( __builtin_types_compatible_p\
        (fp_t, double), 1.0, 1.0f)

// ========================================================================
// Identity
DECLARE_ACTIVATION_FUNCTION(Identity,
{
    return x;
}, {
    return 1;
})

// Linear(x, m)
// Elu(x, Alpha)

DECLARE_ACTIVATION_FUNCTION(ReLU,
{
    return (x > 0) ? x : 0;
}, {
    return (x > 0) ? 1 : 0;
})

// LeakyRelu(x, alpha)

// https://en.wikipedia.org/wiki/Sigmoid_function
DECLARE_ACTIVATION_FUNCTION(Sigmoid,
{
    return 1 / (1 + exp(-x));
}, {
    fp_t y = Sigmoid(x);
    return y * (1 - y);
})

DECLARE_ACTIVATION_FUNCTION(Tanh,
{
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
}, {
    return 1 - pow(Tanh(x), 2);
})

// Exp, GeLU, HardSigmoid, SeLU, softplus, softsign, swish, trelu