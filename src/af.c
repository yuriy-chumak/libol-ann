// activation functions
#include <ann.h>

#include <math.h>

// https://en.wikipedia.org/wiki/Sigmoid_function
__attribute__((used))
word OL_sigmoid(olvm_t* this, word arguments)
{
    word A = car(arguments); arguments = cdr(arguments); // matrix A
    assert (arguments == INULL);

    word B = new_matrix(this,
        matrix_m(A),
        matrix_n(A), &A);

    fp_t* a = matrix_floats(A);
    fp_t* b = matrix_floats(B);

    size_t len = matrix_len(A);
    for (size_t i = 0; i < len; i++) {
        fp_t x = *a++;
        *b++ = 1.0f / (1.0f + exp(-x));
    }

    return B;
}

__attribute__((used))
word OL_sigmoidE(olvm_t* this, word arguments)
{
    word A = car(arguments); arguments = cdr(arguments); // matrix A
    assert (arguments == INULL);

    size_t m = matrix_m(A);
    size_t n = matrix_n(A);

    word B = A; // we do not create new matrix, just change existing

    fp_t* a = matrix_floats(A);
    fp_t* b = matrix_floats(B);

    size_t len = matrix_len(A);
    for (size_t i = 0; i < len; i++) {
        fp_t x = *a++;
        *b++ = 1.0f / (1.0f + exp(-x));
    }

    return B;
}

__attribute__((used))
word* OL_sigmoidD(olvm_t* this, word* arguments)
{
    word* A = (word*)car(arguments); arguments = (word*)cdr(arguments); // matrix A
    assert ((word)arguments == INULL);

    size_t m = value(ref(A, 1));
    size_t n = value(ref(A, 2));

    word* B = (word*)new_matrix(this, m, n, &A);

    float* a = (float*) (ref(A, 3) + W);
    float* b = (float*) (ref(B, 3) + W);

    size_t size = rawstream_size(ref(A, 3)) / sizeof(fp_t);
    for (size_t i = 0; i < size; i++) { // todo: use real 
        fp_t x = *a++;
        fp_t sigm = 1.0f / (1.0f + exp(-x)); // x?
        *b++ = sigm * (1 - sigm);
    }

    return B;
}

__attribute__((used))
word* OL_sigmoidDE(olvm_t* this, word* arguments)
{
    word* A = (word*)car(arguments); arguments = (word*)cdr(arguments); // matrix A
    assert ((word)arguments == INULL);

    size_t m = value(ref(A, 1));
    size_t n = value(ref(A, 2));

    word* B = A; // we do not create new matrix, just change existing

    float* a = (float*) (ref(A, 3) + W);
    float* b = (float*) (ref(B, 3) + W);

    size_t size = rawstream_size(ref(A, 3)) / sizeof(fp_t);
    for (size_t i = 0; i < size; i++) { // todo: use real 
        fp_t x = *a++;
        fp_t sigm = 1.0f / (1.0f + exp(-x));
        *b++ = sigm * (1 - sigm);
    }

    return B;
}
