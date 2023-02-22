#pragma once
#include <ol/vm.h>

// тип, который мы будем использовать для вычислений (обычно - float)
typedef float fp_t;

double OL2D(word arg); float OL2F(word arg); // implemented in olvm.c
#define ol2f(num)\
    __builtin_choose_expr( __builtin_types_compatible_p\
        (fp_t, double), OL2D(num), OL2F(num))

#define matrix_m(M)             ( value(ref(M, 1)) )
#define matrix_n(M)             ( value(ref(M, 2)) )
#define matrix_floats(M) ( (fp_t*) &car(ref(M, 3)) )
#define matrix_len(M)  ( rawstream_size(ref(M, 3)) / sizeof(fp_t) )

#define is_matrix(M) ( is_vector(M) ) //&& (object_size(M) == 4) )

// functions.
#define NARG(...) NARG_N(_, ## __VA_ARGS__,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0)
#define NARG_N(_,n0,n1,n2,n3,n4,n5,n6,n7,n8,n9,mn10,n11,n12,n13,n14,n15,n16,n17,n18, n,...) n
#define new_matrix(this, m, n, ...) new_matrix_(this, m, n, NARG(__VA_ARGS__), ##__VA_ARGS__)
word new_matrix_(olvm_t* this, size_t m, size_t n, size_t nw, ...);

// (mrandom! matrix )
// word OL_mrandomE(olvm_t* this, word* arguments);

// activation functions:

// (sigmoid matrix)
// word OL_sigmoid(olvm_t* this, word arguments);
// word OL_sigmoidE(olvm_t* this, word arguments);

// input/output
// word OL_mwrite(olvm_t* this, word* arguments);
// word OL_mread(olvm_t* this, word* arguments);
// word OL_mreadE(olvm_t* this, word* arguments);
