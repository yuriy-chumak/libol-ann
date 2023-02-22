#include <ann.h>

#include <math.h>
#define min(a,b) ((a) < (b) ? (a) : (b))
#define max(a,b) ((a) > (b) ? (a) : (b))
#define clamp(x, a, b) ({\
    typeof(x) v = x;\
    min(max(x, a), b);\
})

__attribute__((used))
word OL_clamp(olvm_t* this, word arguments)
{
    word A = car(arguments); arguments = cdr(arguments); // matrix A
    word X = car(arguments); arguments = cdr(arguments);
    word Y = car(arguments); arguments = cdr(arguments);
    assert (arguments == INULL);

    word B = new_matrix(this,
        matrix_m(A),
        matrix_n(A), &A);

    fp_t x = ol2f(X);
    fp_t y = ol2f(Y);

    fp_t* a = matrix_floats(A);
    fp_t* b = matrix_floats(B);
    size_t len = matrix_len(A);
    for (size_t i = 0; i < len; i++) {
        *b++ = clamp(*a++, x, y);
    }

    return B;
}

__attribute__((used))
word OL_clampE(olvm_t* this, word arguments)
{
    word A = car(arguments); arguments = cdr(arguments); // matrix A
    word X = car(arguments); arguments = cdr(arguments);
    word Y = car(arguments); arguments = cdr(arguments);
    assert (arguments == INULL);

    word B = A; // we do not create new matrix, just change existing

    fp_t x = ol2f(X);
    fp_t y = ol2f(Y);

    fp_t* a = matrix_floats(A);
    fp_t* b = matrix_floats(B);
    size_t len = matrix_len(A);
    for (size_t i = 0; i < len; i++) {
        *b++ = clamp(*a++, x, y);
    }

    return B;
}
