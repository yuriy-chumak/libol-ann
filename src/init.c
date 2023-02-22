#include <ann.h>

// (mrandom! matrix)
__attribute__((used))
word OL_mrandomE(olvm_t* this, word* arguments)
{
    word A = car(arguments); arguments = (word*)cdr(arguments); // matrix
    assert ((word)arguments == INULL);

    size_t m = matrix_m(A);
    size_t n = matrix_n(A);

    size_t len = matrix_len(A);

    // random();
    fp_t* floats = matrix_floats(A);
    for (int i = 0; i < len; i++)
        *floats++ = (2.0 * (fp_t)rand() / (fp_t)RAND_MAX) - 1.0;

    return A;
}

// (mrandom! matrix)
__attribute__((used))
word OL_zeroE(olvm_t* this, word* arguments)
{
    word A = car(arguments); arguments = (word*)cdr(arguments); // matrix
    assert ((word)arguments == INULL);

    size_t m = matrix_m(A);
    size_t n = matrix_n(A);

    size_t len = matrix_len(A);

    // random();
    fp_t* floats = matrix_floats(A);
    for (int i = 0; i < len; i++)
        *floats++ = 0.0;

    return A;
}

// (mreshape matrix rows columns)
__attribute__((used))
word OL_mreshape(olvm_t* this, word arguments)
{
    word* fp; // this easily indicate that we do manual memory allocations
    heap_t* heap = (struct heap_t*)this;

    word A = car(arguments); arguments = cdr(arguments); // matrix
    word M = car(arguments); arguments = cdr(arguments);
    word N = car(arguments); arguments = cdr(arguments);
    assert (arguments == INULL);

    size_t m = matrix_m(A);
    size_t n = matrix_n(A);

    int nm = number(M);
    int nn = number(N);

    if (m*n != nm*nn)
        return IFALSE;

    fp = heap->fp;
    word object = (word) new_vector(M, N, ref(A, 3));
    heap->fp = fp;

    return object;
}
