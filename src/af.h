#pragma once
#include <ann.h>

#define DECLARE_FOREACH(fnc) \
__attribute__((visibility("hidden"))) \
void fnc ## _foreach(word A, word B, size_t len)\
{\
    fp_t* a = matrix_floats(A);\
    fp_t* b = matrix_floats(B);\
\
    for (size_t i = 0; i < len; i++) {\
        fp_t x = *a++;\
        *b++ = fnc(x);\
    }\
}

#define DECLARE_OL_(name) \
__attribute__((used)) /* activation function */ \
word OL_ ## name(olvm_t* this, word arguments)\
{\
    word A = car(arguments); arguments = cdr(arguments);\
    assert (arguments == INULL);\
\
    word B = new_matrix(this,\
        matrix_m(A),\
        matrix_n(A), &A);\
\
    name ## _foreach(A, B, matrix_len(A));\
    return B;\
}\
__attribute__((used)) /* activation function ! */ \
word OL_ ## name ## E(olvm_t* this, word arguments)\
{\
    word A = car(arguments); arguments = cdr(arguments);\
    assert (arguments == INULL);\
\
    word B = A; /* we do not create new matrix, just change existing */ \
\
    name ## _foreach(A, B, matrix_len(A));\
    return B;\
}


// main macro
#define DECLARE_ACTIVATION_FUNCTION(name, function, prime)\
    __inline__ \
    fp_t name(fp_t x) function \
    DECLARE_FOREACH(name) \
    DECLARE_OL_(name) \
    __inline__ \
    fp_t name ## _prime(fp_t x) prime \
    DECLARE_FOREACH(name ## _prime) \
    DECLARE_OL_(name ## _prime)

