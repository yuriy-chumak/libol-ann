#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

#include <math.h>
#include <stdio.h>

#include <ann.h>

// наши вектора - это матрицы из одной строки(!)

// TODO: use https://github.com/hfp/libxsmm/blob/master/samples/hello/hello.c
// TODO: Matrix Multiplication
//       https://cnugteren.github.io/tutorial/pages/page1.html

// ---------------------------------------------------------
// эта функция не только создает новую матрицу, но и
// 1. вызывает GC, если для матрицы не хватает места
// 2. сохраняет и восстанавливает Ol-объекты, если они были
//    перемещены GC

__attribute__((visibility("hidden")))
word new_matrix_(olvm_t* this, size_t m, size_t n, size_t nw, ...)
{
    word* fp;  // индикатор того, что используем выделение памяти
    heap_t* heap = (struct heap_t*)this;
    size_t msize = m * n * sizeof(fp_t);

    size_t words = (msize + (W-1)) / W;
    if ((heap->fp + words) > heap->end) {
        va_list ptrs;
        size_t p[nw];

        // save Ol objects before GC
        va_start(ptrs, nw);
        for (int i = 0; i < nw; i++)
            p[i] = OLVM_pin(this, *(va_arg(ptrs, word*)));
        va_end(ptrs);

        heap->gc(this, words);

        // restore OL objects after GC
        va_start(ptrs, nw);
        for (int i = 0; i < nw; i++)
            *(va_arg(ptrs, word*)) = OLVM_unpin((struct olvm_t*)this, p[i]);
        va_end(ptrs);
    }

    fp = heap->fp;
    word floats = (word) new_bytevector(msize);
    word matrix = (word) new_vector(I(m), I(n), floats);
    heap->fp = fp;
    return matrix;
}


// (mnew rows columns)
__attribute__((used))
word OL_mnew(olvm_t* this, word arguments)
{
    word A = car(arguments); arguments = cdr(arguments); // m
    word B = car(arguments); arguments = cdr(arguments); // n
    assert (arguments == INULL);

    size_t m = value(A);
    size_t n = value(B);

    return new_matrix(this, m, n);
}

// (at matrix row column)
// индексация начинается с 1, негитивные индексы поддерживаются
__attribute__((used))
word OL_mref(olvm_t* this, word arguments)
{
    word* fp; // this easily indicate that we do manual memory allocations
    heap_t* heap = (struct heap_t*)this;

    word A = car(arguments); arguments = cdr(arguments); // matrix
    word I = car(arguments); arguments = cdr(arguments); // (i) row
    word J = car(arguments); arguments = cdr(arguments); // (j) column
    assert (arguments == INULL);

    // размерность
    uint32_t m = matrix_m(A);
    uint32_t n = matrix_n(A);

    // мы поддерживаем и негативные индексы тоже
    int i = number(I); if (i < 0) i += m;
    int j = number(J); if (j < 0) j += n;

    // проверка индексации
    if (i == 0 || i > m || j == 0 || j > n)
        return IFALSE;
    --i; --j; // приводим индексацию к C

    size_t len = matrix_len(A);
    if (i*n + j >= len)  // invalid matrix?
        return IFALSE;

    // работаем
    fp_t* floats = matrix_floats(A);
    fp_t v = floats[i*n + j];

    fp = heap->fp;
    word object = new_inexact(v);
    heap->fp = fp;

    return object;
}

// (msetref! matrix i j value)
__attribute__((used))
word OL_setrefE(olvm_t* this, word arguments)
{
    word A = car(arguments); arguments = cdr(arguments);
    word I = car(arguments); arguments = cdr(arguments); // i
    word J = car(arguments); arguments = cdr(arguments); // j
    word X = car(arguments); arguments = cdr(arguments); // value
    assert (arguments == INULL);

    size_t m = matrix_m(A); // rows
    size_t n = matrix_n(A); // columns

    // мы поддерживаем и негативные индексы тоже
    int i = number(I); if (i < 0) i += m;
    int j = number(J); if (j < 0) j += n;

    // проверка индексации
    if (i == 0 || i > m || j == 0 || j > n)
        return IFALSE;
    --i; --j; // приводим индексацию к C

    size_t len = matrix_len(A);
    if (i*n + j >= len)  // invalid matrix?
        return IFALSE;

    // работаем
    fp_t x = ol2f(X);

    fp_t* floats = matrix_floats(A);
    floats[i*n + j] = x;
    return X;
}

// (bytevector->matrix bytevector)
// (bytevector->matrix bytevector scale)
__attribute__((used))
word OL_bv2m(olvm_t* this, word arguments)
{
    fp_t ds = 1.0f; // downscale

    word A = car(arguments); arguments = cdr(arguments); // bytevector
    if (arguments != INULL) {
        word B = car(arguments); arguments = cdr(arguments); // downscale (todo: process ALL possible numbers)
        ds = ol2f(B);
    }
    assert (arguments == INULL);

    // todo: сохранить матрицу
    size_t n = rawstream_size(A);
    word matrix = new_matrix(this, 1, n, &A);

    fp_t* f = matrix_floats(matrix);
    fp_t* a = (fp_t*) &car(A);
    for (size_t j = 0; j < n; j++)
        *f++ = (fp_t)*a++ / ds;

    return matrix;
}

// (vector->matrix vector)
// (vector->matrix vector scale)
__attribute__((used))
word OL_v2m(olvm_t* this, word arguments)
{
    fp_t ds = 1.0f; // default downscale

    word A = car(arguments); arguments = cdr(arguments); // vector
    if (arguments != INULL) {
        word B = car(arguments); arguments = cdr(arguments);
        ds = ol2f(B);
    }
    assert (arguments == INULL);

    size_t n = reference_size(A);
    word matrix = new_matrix(this, 1, n, &A);

    fp_t* f = matrix_floats(matrix);
    for (size_t j = 1; j <= n; j++)
        *f++ = ol2f(ref(A, j)) / ds;

    return matrix;
}

// (list->matrix list)
// (list->matrix list scale)
__attribute__((used))
word OL_l2m(olvm_t* this, word arguments)
{
    fp_t ds = 1.0f; // downscale

    word A = car(arguments); arguments = cdr(arguments); // list
    if (arguments != INULL) {
        word B = car(arguments); arguments = cdr(arguments); // downscale
        ds = ol2f(B);
    }
    assert (arguments == INULL);

    size_t n = 0;
    word p = A;
    while (p != INULL) {
        n++;
        p = cdr(p);
    }
    word matrix = new_matrix(this, 1, n, &A);

    fp_t* f = matrix_floats(matrix);
    while (A != INULL) {
        fp_t num = ol2f(car(A));
        *f++ = num / ds;
        A = cdr(A);
    }

    return matrix;
}

// // (matrix->list matrix)
// __attribute__((used))
// word OL_f2l(olvm_t* this, word arguments)
// {
//     word* fp;
//     heap_t* heap = (struct heap_t*) this;

//     word A = car(arguments); arguments = cdr(arguments); // matrix
//     assert (arguments == INULL);

//     size_t m = matrix_m(A);
//     size_t n = matrix_n(A);
//     fp_t* a = matrix_floats(A);

//     fp = heap->fp;
//     word p = INULL;
//     for (int i = (m * n) - 1; i >= 0; i--)
//         p = (word) new_pair(new_inexact(a[i]), p);
//     heap->fp = fp;

//     return p;
// }


// // todo: move to top (or different file 'conv.c' or similar name)

// __attribute__((used))
// word* OL_abs(olvm_t* this, word* arguments)
// {
//     word* A = (word*)car(arguments); arguments = (word*)cdr(arguments); // matrix A
//     assert ((word)arguments == INULL);

//     size_t m = value(ref(A, 1));
//     size_t n = value(ref(A, 2));

//     word* B = (word*)new_matrix(this, m, n, &A);

//     float* a = (float*) (ref(A, 3) + W);
//     float* b = (float*) (ref(B, 3) + W);

//     size_t size = rawstream_size(ref(A, 3)) / sizeof(fp_t);
//     for (size_t i = 0; i < size; i++) { // todo: use real 
//         fp_t x = *a++;
//         fp_t f = (fp_t)fabs(x);
//         *b++ = f;
//     }

//     return B;
// }


// __attribute__((used))
// word* OL_mean(olvm_t* this, word* arguments)
// {
//     word* fp;
//     heap_t* heap = (struct heap_t*)this;

//     word* A = (word*)car(arguments); arguments = (word*)cdr(arguments); // matrix A
//     assert ((word)arguments == INULL);

//     size_t m = value(ref(A, 1));
//     size_t n = value(ref(A, 2));

//     float* a = (float*) (ref(A, 3) + W);

//     size_t size = rawstream_size(ref(A, 3)) / sizeof(fp_t);
//     inexact_t S = 0;
//     for (size_t i = 0; i < size; i++) { // todo: use real 
//         S += *a++;
//     }
//     S /= (inexact_t)size;

//     fp = heap->fp;
//     word* object = (word*) new_inexact(S);
//     heap->fp = fp;

//     return object;
// }

__attribute__((used))
word OL_dot(olvm_t* this, word arguments)
{
    word A = car(arguments); arguments = cdr(arguments); // matrix A
    word B = car(arguments); arguments = cdr(arguments); // matrix B
    assert (arguments == INULL);

    size_t m = value(ref(A, 1));
    size_t n = value(ref(A, 2));

    if (value(ref(B, 1)) != n)
        return IFALSE;

    size_t q = value(ref(B, 2));

    word C = new_matrix(this, m, q, &A, &B);

    fp_t* a = (fp_t*) (ref(A, 3) + W);
    fp_t* b = (fp_t*) (ref(B, 3) + W);
    fp_t* c = (fp_t*) (ref(C, 3) + W);

    size_t i,j,k;
#pragma omp parallel shared(a,b,c) private(i,j)
    for (i = 0; i < m; i++) {
        for (j = 0; j < q; j++) {
            fp_t S = 0;
#pragma omp parallel shared(a,b,c) firstprivate(i,j) private(k) reduction(+:S)
            for (k = 0; k < n; k++) {
                fp_t f1 = a[i*n + k];
                fp_t f2 = b[k*q + j];
                S += a[i*n + k] * b[k*q + j];
            }

            c[i*q + j] = S;
        }
    }

    return C;
}

__attribute__((used))
word* OL_sub(olvm_t* this, word* arguments)
{
    word* A = (word*)car(arguments); arguments = (word*)cdr(arguments); // matrix A
    word* B = (word*)car(arguments); arguments = (word*)cdr(arguments); // matrix B
    assert ((word)arguments == INULL);
    // todo: assert for matrix sizes

    size_t m = value(ref(A, 1));
    size_t n = value(ref(A, 2));

    if (value(ref(B, 1)) != m)
        return (word*)IFALSE;
    if (value(ref(B, 2)) != n)
        return (word*)IFALSE;

    word* C = (word*)new_matrix(this, m, n, &A, &B);

    float* a = (float*) (ref(A, 3) + W);
    float* b = (float*) (ref(B, 3) + W);
    float* c = (float*) (ref(C, 3) + W);

    size_t size = rawstream_size(ref(A, 3)) / sizeof(fp_t); // todo: assert for matrix B size
    for (size_t i = 0; i < size; i++)
        *c++ = *a++ - *b++;

    return C;
}

// __attribute__((used))
// word* OL_add(olvm_t* this, word* arguments)
// {
//     word* A = (word*)car(arguments); arguments = (word*)cdr(arguments); // matrix A
//     word* B = (word*)car(arguments); arguments = (word*)cdr(arguments); // matrix B
//     assert ((word)arguments == INULL);
//     // todo: assert for matrix sizes

//     size_t m = value(ref(A, 1));
//     size_t n = value(ref(A, 2));

//     if (value(ref(B, 1)) != m)
//         return (word*)IFALSE;
//     if (value(ref(B, 2)) != n)
//         return (word*)IFALSE;

//     word* C = (word*)new_matrix(this, m, n, &A, &B);

//     float* a = (float*) (ref(A, 3) + W);
//     float* b = (float*) (ref(B, 3) + W);
//     float* c = (float*) (ref(C, 3) + W);

//     size_t size = rawstream_size(ref(A, 3)) / sizeof(fp_t); // todo: assert for matrix B size
//     for (size_t i = 0; i < size; i++)
//         *c++ = *a++ + *b++;

//     return C;
// }

__attribute__((used))
word* OL_addE(olvm_t* this, word* arguments)
{
    word* A = (word*)car(arguments); arguments = (word*)cdr(arguments); // matrix A
    word* B = (word*)car(arguments); arguments = (word*)cdr(arguments); // matrix B
    assert ((word)arguments == INULL);
    // todo: assert for matrix sizes

    size_t m = value(ref(A, 1));
    size_t n = value(ref(A, 2));

    if (value(ref(B, 1)) != m)
        return (word*)IFALSE;
    if (value(ref(B, 2)) != n)
        return (word*)IFALSE;

    float* a = (float*) (ref(A, 3) + W);
    float* b = (float*) (ref(B, 3) + W);

    size_t size = rawstream_size(ref(A, 3)) / sizeof(fp_t); // todo: assert for matrix B size
    for (size_t i = 0; i < size; i++)
        *a++ += *b++;

    return A;
}

// (mul matrix matrix) = cartesian product
// (mul matrix scalar)
__attribute__((used))
word OL_mul(olvm_t* this, word arguments)
{
    word A = car(arguments); arguments = cdr(arguments); // A matrix
    word B = car(arguments); arguments = cdr(arguments); // B matrix or scalar
    assert ((word)arguments == INULL);
    // todo: assert for matrix sizes

    size_t m = value(ref(A, 1));
    size_t n = value(ref(A, 2));

    if (is_matrix(B)) {
        if (value(ref(B, 1)) != m)
            return IFALSE;
        if (value(ref(B, 2)) != n)
            return IFALSE;
    }

    word C = new_matrix(this, m, n, &A, &B);

    float* a = (float*) (ref(A, 3) + W);
    float* c = (float*) (ref(C, 3) + W);

    size_t size = rawstream_size(ref(A, 3)) / sizeof(fp_t);
    if (is_matrix(B)) {
        float* b = (float*) (ref(B, 3) + W);
        for (size_t i = 0; i < size; i++)
            *c++ = *a++ * *b++;
    }
    else {
        fp_t b = ol2f(B);
        for (size_t i = 0; i < size; i++)
            *c++ = *a++ * b;
    }

    return C;
}

__attribute__((used))
word* OL_T(olvm_t* this, word* arguments) // transpose
{
    word A = car(arguments); arguments = (word*)cdr(arguments); // matrix A
    assert ((word)arguments == INULL);

    size_t m = matrix_m(A);
    size_t n = matrix_n(A);

    word*B = (word*)new_matrix(this, n, m, &A);

    fp_t* a = matrix_floats(A);
    fp_t* b = matrix_floats(B);

    for (size_t i = 0; i < m; i++)
        for (size_t j = 0; j < n; j++)
            b[j*m + i] = a[i*n + j];

    return B;
}


// // OLD: ==================================================
// // typedef struct _matrix_t
// // {
// //     size_t rows;
// //     size_t columns;
// //     float* floats;
// // } matrix_t;

// // // create new float matrix m*n
// // matrix_t* new(size_t m, size_t n)
// // {
// //     matrix_t* matrix = (matrix_t*)malloc(sizeof(matrix_t));
// //     matrix->rows = m;
// //     matrix->columns = n;
// //     matrix->floats = (float*)malloc(m * n * sizeof(float));
// // }

// // void ann_random(matrix_t* matrix)
// // {
// //     float* x = matrix->floats;
// //     size_t m = matrix->rows;
// //     size_t n = matrix->columns;
// //     for (size_t i = 0; i < m * n; i++)
// //         x[i] = (2.0 * (float)rand() / (float)RAND_MAX) - 1.0;
// // }

// // void ann_read(matrix_t* matrix, char* filename)
// // {
// //     FILE* f = fopen(filename, "rb");
// //     size_t todo = matrix->rows * matrix->columns;
// //     size_t read;
// //     if (f != NULL) {
// //         read = fread(matrix->floats, sizeof(float), todo, f);
// //         fclose(f);
// //     }
// // }

// // void ann_write(matrix_t* matrix, char* filename)
// // {
// //     FILE* f = fopen(filename, "wb");
// //     size_t todo = matrix->rows * matrix->columns;
// //     if (f != NULL) {
// //         fwrite(matrix->floats, sizeof(float), todo, f);
// //         fclose(f);
// //     }
// // }

// // void dot(matrix_t* A /* m*n */, matrix_t* B /* n*q */, matrix_t* C /* m*q */)
// // {
// //     size_t m = A->rows;
// //     size_t n = A->columns;
// //     size_t q = B->columns;

// //     assert(B->rows == n);
// //     assert(C->rows == m);
// //     assert(C->columns == q);

// //     float* a = A->floats;
// //     float* b = B->floats;
// //     float* c = C->floats;

// //     for (size_t i = 0; i < m; i++) {
// //         for (size_t j = 0; j < q; j++) {
// //             c[i + j*m] = 0;
// //             for(size_t k = 0; k < n; k++)
// //                 c[i + j*m] += a[i + k*m] * b[k + j*n];
// //         }
// //     }
// // }

// // float at(matrix_t* matrix, size_t i, size_t j) {
// //     assert(i <= matrix->rows);
// //     assert(j <= matrix->columns);
// //     return matrix->floats[i + j*matrix->rows];
// // }

// // float at1(matrix_t* matrix, size_t i, size_t j) {
// //     assert(i <= matrix->rows);
// //     assert(j <= matrix->columns);
// //     return matrix->floats[(i-1) + (j-1)*matrix->rows];
// // }

// // // transpose
// // void T(matrix_t* A, matrix_t* B)
// // {
// //     assert(A->rows == B->rows);
// //     assert(A->columns == B->columns);
// //     size_t m = A->rows;
// //     size_t n = A->columns;
// //     float* a = A->floats;
// //     float* b = B->floats;

// //     for (size_t i = 0; i < m; i++) {
// //         for (size_t j = 0; j < n; j++) {
// //             b[i + j*n] = a[j + i*n];
// //         }
// //     }
// // }

// // void add(matrix_t* A, matrix_t* B, matrix_t* C)
// // {
// //     assert(A->rows == B->rows);
// //     assert(A->columns == B->columns);
// //     assert(A->rows == C->rows);
// //     assert(A->columns == C->columns);
// //     size_t m = A->rows;
// //     size_t n = A->columns;
// //     float* a = A->floats;
// //     float* b = B->floats;
// //     float* c = C->floats;

// //     for (size_t i = 0; i < m; i++) {
// //         for (size_t j = 0; j < n; j++) {
// //             c[i + j*n] = a[i + j*n] + b[i + j*n];
// //         }
// //     }
// // }

// // void mul(matrix_t* A, matrix_t* B, matrix_t* C)
// // {
// //     assert(A->rows == B->rows);
// //     assert(A->columns == B->columns);
// //     assert(A->rows == C->rows);
// //     assert(A->columns == C->columns);
// //     size_t m = A->rows;
// //     size_t n = A->columns;
// //     float* a = A->floats;
// //     float* b = B->floats;
// //     float* c = C->floats;

// //     for (size_t i = 0; i < m; i++) {
// //         for (size_t j = 0; j < n; j++) {
// //             c[i + j*n] = a[i + j*n] * b[i + j*n];
// //         }
// //     }
// // }

// // void rsub1(matrix_t* A, matrix_t* B)
// // {
// //     assert(A->rows == B->rows);
// //     assert(A->columns == B->columns);
// //     size_t m = A->rows;
// //     size_t n = A->columns;
// //     float* a = A->floats;
// //     float* b = B->floats;

// //     for (size_t i = 0; i < m; i++) {
// //         for (size_t j = 0; j < n; j++) {
// //             b[i + j*n] = 1.0f - a[i + j*n];
// //         }
// //     }
// // }

// // void sigmoid(matrix_t* A, matrix_t* B)
// // {
// //     assert(A->rows == B->rows);
// //     assert(A->columns == B->columns);
// //     size_t m = A->rows;
// //     size_t n = A->columns;
// //     float* a = A->floats;
// //     float* b = B->floats;

// //     for (size_t i = 0; i < m; i++) {
// //         for (size_t j = 0; j < n; j++) {
// //             float x = a[i + j*n];
// //             b[i + j*n] = 1.0f / (1.0f + exp(0.0f - x));
// //         }
// //     }
// // }

// // void sigmoid_d(matrix_t* A, matrix_t* B)
// // {
// //     assert(A->rows == B->rows);
// //     assert(A->columns == B->columns);
// //     size_t m = A->rows;
// //     size_t n = A->columns;
// //     float* a = A->floats;
// //     float* b = B->floats;

// //     for (size_t i = 0; i < m; i++) {
// //         for (size_t j = 0; j < n; j++) {
// //             float x = a[i + j*n];
// //             b[i + j*n] = (x * (1.0f - x));
// //         }
// //     }
// // }

// // void ann_set(matrix_t*A, float* f)
// // {
// //     size_t m = A->rows;
// //     size_t n = A->columns;
// //     float* a = A->floats;
// //     size_t i = m*n;
// //     while (i--)
// //         *a++ = *f++;
// // }

// // /*
// // ;; (define ann (load-dynamic-library "libann.so"))
// // ;; (define matrix_t* fft-void*)
// // ;; (define size_t fft-int)
// // ;; (define void fft-void)
// // ;; (define new (ann matrix_t* "new" size_t size_t))
// // ;; (define ann-random (ann void "ann_random" matrix_t*))
// // ;; (define ann-dot (ann void "dot" matrix_t* matrix_t* matrix_t*))
// // ;; (define ann-read (ann void "ann_read" matrix_t* type-string))
// // ;; (define ann-write (ann void "ann_write" matrix_t* type-string))
// // ;; (define ann-set (ann void "ann_set" matrix_t* (fft* fft-float)))
// // ;; (define ann-sigmoid (ann void "sigmoid" matrix_t* matrix_t*))
// // ;; (define ann-at (ann fft-float "at" matrix_t* size_t size_t))
// // */