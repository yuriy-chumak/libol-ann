#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

#include <math.h>
#include <stdio.h>

#include <ol/vm.h>

// тип, который мы будем использовать для вычислений (обычно - float)
typedef float fp_t;

#define MEMPAD (1024) // TODO: exclude this

// TODO: use https://github.com/hfp/libxsmm/blob/master/samples/hello/hello.c
// TODO: Matrix Multiplication
//       https://cnugteren.github.io/tutorial/pages/page1.html

// ---------------------------------------------------------
// эта функция не только создает новую матрицу, но и
// 1. вызывает GC, если для матрицы нехватает места
// 2. сохраняет и восстанавливает Ol-объекты, если они были
//    перемещены GC
static
word* create_new_matrix_(olvm_t* this, size_t m, size_t n, size_t nw, ...)
{
    word* fp;
	heap_t* heap = (struct heap_t*)this;
    size_t vsize = m * n * sizeof(fp_t);

    size_t words = (vsize + (W-1)) / W;
    if ((heap->fp + words) > (heap->end - MEMPAD)) {
        va_list ptrs;
        size_t id[nw];

        // save Ol objects before GC
		va_start(ptrs, nw);
        for (int i = 0; i < nw; i++)
            id[i] = OLVM_pin(this, *(va_arg(ptrs, word*)));
        va_end(ptrs);

        heap->gc(this, words);

		// restore OL objects after GC
        va_start(ptrs, nw);
        for (int i = 0; i < nw; i++)
            *(va_arg(ptrs, word*)) = OLVM_unpin((struct olvm_t*)this, id[i]);
        va_end(ptrs);
    }

    fp = heap->fp;
	word* floats = new_bytevector(vsize);
    word* matrix = new_vector(I(m), I(n), floats);
    heap->fp = fp;
    return matrix;
}

#define NARG(...) NARG_N(_, ## __VA_ARGS__,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0)
#define NARG_N(_,n0,n1,n2,n3,n4,n5,n6,n7,n8,n9,mn10,n11,n12,n13,n14,n15,n16,n17,n18, n,...) n
#define create_new_matrix(this, m, n, ...) create_new_matrix_(this, m, n, NARG(__VA_ARGS__), ##__VA_ARGS__)


//PUBLIC
__attribute__((used))
word* OL_mnew(olvm_t* this, word* arguments)
{
    word* A = (word*)car(arguments); arguments = (word*)cdr(arguments); // m
    word* B = (word*)car(arguments); arguments = (word*)cdr(arguments); // n
    assert ((word)arguments == INULL);

    size_t m = value(A);
    size_t n = value(B);

    word* matrix = create_new_matrix(this, m, n);
    return matrix;
}

__attribute__((used))
word* OL_at(olvm_t* this, word* arguments)
{
    word* fp; // this easily indicate that we do manual memory allocations
	heap_t* heap = (struct heap_t*)this;

    word* A = (word*)car(arguments); arguments = (word*)cdr(arguments); // matrix
    word* R = (word*)car(arguments); arguments = (word*)cdr(arguments); // (i) row
    word* C = (word*)car(arguments); arguments = (word*)cdr(arguments); // (j) column
    assert ((word)arguments == INULL);

    // размерность
    uint32_t m = value(ref(A, 1));
    uint32_t n = value(ref(A, 2));

    size_t i = value(R);
    size_t j = value(C);

    size_t size = rawstream_size(ref(A, 3)) / sizeof(fp_t);
    if (i == 0 || i > m ||
        i == 0 || j > n ||
        ((i-1)*n + (j-1)) >= size) { // invalid matrix
        return (word*)IFALSE;
    }
    --i; --j;

    fp = heap->fp;
    word* object = new_rawstream(TINEXACT, sizeof(inexact_t));
    heap->fp = fp;

    fp_t* floats = (fp_t*) (ref(A, 3) + W);
    fp_t v = floats[i*n + j];
    *(inexact_t*)(object + 1) = v;

    return object;
}


__attribute__((used))
word* OL_mrandomE(olvm_t* this, word* arguments)
{
    word* A = (word*)car(arguments); arguments = (word*)cdr(arguments); // matrix
    assert ((word)arguments == INULL);

    size_t m = value(ref(A, 1));
    size_t n = value(ref(A, 2));

    size_t size = rawstream_size(ref(A, 3)) / sizeof(fp_t);

    // random();
    fp_t* floats = (float*) (ref(A, 3) + W);
    for (int i = 0; i < size; i++)
        *floats++ = (2.0 * (fp_t)rand() / (fp_t)RAND_MAX) - 1.0;

    return A;
}

__attribute__((used))
word* OL_mwrite(olvm_t* this, word* arguments)
{
    word* M = (word*)car(arguments); arguments = (word*)cdr(arguments); // matrix
    word* A = (word*)car(arguments); arguments = (word*)cdr(arguments); // filename
    assert ((word)arguments == INULL);

    size_t flen = rawstream_size(A);
    char* filename = __builtin_alloca(flen+1);
    memcpy(filename, &car(A), flen);
    filename[flen] = 0;

    FILE* file = fopen(filename, "wb");
    if (!file)
        return (word*)IFALSE;

    // заголовок (1 в конце значит что это флоаты)
    int32_t magic = *(int32_t*)"ann\x1";

    fwrite(&magic, sizeof(magic), 1, file);

    // размерность
    uint32_t m = value(ref(M, 1));
    uint32_t n = value(ref(M, 2));

    fwrite(&m, sizeof(m), 1, file);
    fwrite(&n, sizeof(n), 1, file);

    // матрица
    float* f = (float*)(ref(M, 3) + W);

    fwrite(f, sizeof(float), m*n, file);

    fclose(file);
    return (word*)ITRUE;
}

__attribute__((used))
word* OL_mread(olvm_t* this, word* arguments)
{
    word* A = (word*)car(arguments); arguments = (word*)cdr(arguments); // matrix
    assert ((word)arguments == INULL);

    size_t flen = rawstream_size(A);
    char* filename = __builtin_alloca(flen+1);
    memcpy(filename, &car(A), flen);
    filename[flen] = 0;

    FILE* file = fopen(filename, "rb");
    if (!file)
        return (word*)IFALSE;

    // заголовок (1 в конце значит, что это флоаты)
    int32_t magic = 0; // = *(int32_t*)"ann\x1";
    size_t read;

    read = fread(&magic, sizeof(magic), 1, file);
    if (magic != *(int32_t*)"ann\x1")
        goto fail;

    // размерность
    uint32_t m = 0; //value(ref(M, 1));
    uint32_t n = 0; //value(ref(M, 2));

    read = fread(&m, sizeof(m), 1, file);
    read = fread(&n, sizeof(n), 1, file);

    if (m == 0 || n == 0)
        goto fail;

    // матрица
    word* matrix = create_new_matrix(this, m, n);

    read = fread((float*)(ref(matrix, 3) + W), sizeof(float), m*n, file);

    fclose(file);
    return matrix;
fail:
    fclose(file);
    return (word*)IFALSE;
}

__attribute__((used)) // todo: change to matrix [1xN] builer, not a floats bytevector?
word* OL_bv2f(olvm_t* this, word* arguments)
{
    fp_t ds = 1.0f; // downscale

    word* A = (word*)car(arguments); arguments = (word*)cdr(arguments); // bytevector
    if (arguments != (word*)INULL) {
        word* B = (word*)car(arguments); arguments = (word*)cdr(arguments); // downscale (todo: process ALL possible numbers)
        ds = number(B); // todo: check this
    }
    assert ((word)arguments == INULL);

    // todo: сохранить матрицу
    size_t n = rawstream_size(A);
    word* matrix = create_new_matrix(this, 1, n, &A);

    fp_t* f = (fp_t*) (ref(matrix, 3) + W);
    unsigned char* a = (unsigned char*) (A + 1);
    for (size_t j = 0; j < n; j++)
        *f++ = (fp_t)*a++ / ds;

    return matrix;
}

__attribute__((used))
word* OL_l2f(olvm_t* this, word* arguments)
{
    float ds = 1.0f; // downscale

    word* A = (word*)car(arguments); arguments = (word*)cdr(arguments); // list
    if (arguments != (word*)INULL) {
        word* B = (word*)car(arguments); arguments = (word*)cdr(arguments); // downscale (нормализация)
        ds = number(B); // same as above
    }
    assert ((word)arguments == INULL);

    size_t n = 0;
    word p = (word)A;
    while (p != INULL) {
        n++;
        p = cdr(p);
    }
    word* matrix = create_new_matrix(this, 1, n, &A);

    float* f = (float*)(ref(matrix, 3) + W);
    p = (word) A;
    while (p != INULL) {
        long num = number(car(p));
        *f++ = (float) num / ds;
        p = cdr(p);
    }

    return matrix;
}

__attribute__((used))
word* OL_f2l(olvm_t* this, word* arguments)
{
	word* fp;
	heap_t* heap = (struct heap_t*)this;

    word* A = (word*)car(arguments); arguments = (word*)cdr(arguments); // matrix
    assert ((word)arguments == INULL);

    size_t m = value(ref(A, 1));
    size_t n = value(ref(A, 2));
    float* floats = (float*)(ref(A, 3) + W);

    fp = heap->fp;
    word *p = (word*)INULL;
    for (int i = (m * n) - 1; i >= 0; i--) {
        word* o = new_rawstream(TINEXACT, sizeof(inexact_t));
        inexact_t* f = (inexact_t*)(o + 1);
        *f = floats[i];
        p = NEW_PAIR(o, p);
    }
    heap->fp = fp;

    return p;
}

__attribute__((used))
word* OL_setrefE(olvm_t* this, word* arguments)
{
    word* A = (word*)car(arguments); arguments = (word*)cdr(arguments);
    word* I = (word*)car(arguments); arguments = (word*)cdr(arguments); // downscale (нормализация)
    word* J = (word*)car(arguments); arguments = (word*)cdr(arguments); // downscale (нормализация)
    word* X = (word*)car(arguments); arguments = (word*)cdr(arguments); // downscale (нормализация)
    assert ((word)arguments == INULL);

    size_t m = value(ref(A, 1));
    size_t n = value(ref(A, 2));

	// printf("m: %d, n: %d\n", m, n);
	return RFALSE;

    size_t i = number(I);
	size_t j = number(J);

    float x = *(float*)(X + 1);

    float* floats = (float*)(ref(A, 3) + W);

	if (i == 0 || i > m || j == 0 || j > n)
		return RFALSE;

	floats[i*n + j] = x;
	return X;
}

// функции активации
// https://en.wikipedia.org/wiki/Sigmoid_function
__attribute__((used))
word* OL_sigmoid(olvm_t* this, word* arguments)
// todo: add "sigmoid! == OL_sigmoidE" version that is not require reallocation
{
    word* A = (word*)car(arguments); arguments = (word*)cdr(arguments); // matrix A
    assert ((word)arguments == INULL);

    size_t m = value(ref(A, 1));
    size_t n = value(ref(A, 2));

    word* B = create_new_matrix(this, m, n, &A);

    float* a = (float*) (ref(A, 3) + W);
    float* b = (float*) (ref(B, 3) + W);

    size_t size = rawstream_size(ref(A, 3)) / sizeof(fp_t);
    for (size_t i = 0; i < size; i++) { // todo: use real 
        fp_t x = *a++;
        *b++ = 1.0f / (1.0f + exp(-x));
    }

    return B;
}

__attribute__((used))
word* OL_sigmoidE(olvm_t* this, word* arguments)
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

    word* B = create_new_matrix(this, m, n, &A);

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

__attribute__((used))
word* OL_abs(olvm_t* this, word* arguments)
{
    word* A = (word*)car(arguments); arguments = (word*)cdr(arguments); // matrix A
    assert ((word)arguments == INULL);

    size_t m = value(ref(A, 1));
    size_t n = value(ref(A, 2));

    word* B = create_new_matrix(this, m, n, &A);

    float* a = (float*) (ref(A, 3) + W);
    float* b = (float*) (ref(B, 3) + W);

    size_t size = rawstream_size(ref(A, 3)) / sizeof(fp_t);
    for (size_t i = 0; i < size; i++) { // todo: use real 
        fp_t x = *a++;
        fp_t f = (fp_t)fabs(x);
        *b++ = f;
    }

    return B;
}


__attribute__((used))
word* OL_mean(olvm_t* this, word* arguments)
{
    word* fp;
	heap_t* heap = (struct heap_t*)this;

    word* A = (word*)car(arguments); arguments = (word*)cdr(arguments); // matrix A
    assert ((word)arguments == INULL);

    size_t m = value(ref(A, 1));
    size_t n = value(ref(A, 2));

    float* a = (float*) (ref(A, 3) + W);

    size_t size = rawstream_size(ref(A, 3)) / sizeof(fp_t);
    inexact_t S = 0;
    for (size_t i = 0; i < size; i++) { // todo: use real 
        S += *a++;
    }
    S /= (inexact_t)size;

    fp = heap->fp;
    word* object = new_rawstream(TINEXACT, sizeof(inexact_t));
    heap->fp = fp;

    *(inexact_t*)(object + 1) = S;

    return object;
}

__attribute__((used))
word* OL_dot(olvm_t* this, word* arguments)
{
    word* A = (word*)car(arguments); arguments = (word*)cdr(arguments); // matrix A
    word* B = (word*)car(arguments); arguments = (word*)cdr(arguments); // matrix B
    assert ((word)arguments == INULL);

    size_t m = value(ref(A, 1));
    size_t n = value(ref(A, 2));

    if (value(ref(B, 1)) != n)
        return (word*)IFALSE;

    size_t q = value(ref(B, 2));

    word* C = create_new_matrix(this, m, q, &A, &B);

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

    word* C = create_new_matrix(this, m, n, &A, &B);

    float* a = (float*) (ref(A, 3) + W);
    float* b = (float*) (ref(B, 3) + W);
    float* c = (float*) (ref(C, 3) + W);

    size_t size = rawstream_size(ref(A, 3)) / sizeof(fp_t); // todo: assert for matrix B size
    for (size_t i = 0; i < size; i++)
        *c++ = *a++ - *b++;

    return C;
}

__attribute__((used))
word* OL_add(olvm_t* this, word* arguments)
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

    word* C = create_new_matrix(this, m, n, &A, &B);

    float* a = (float*) (ref(A, 3) + W);
    float* b = (float*) (ref(B, 3) + W);
    float* c = (float*) (ref(C, 3) + W);

    size_t size = rawstream_size(ref(A, 3)) / sizeof(fp_t); // todo: assert for matrix B size
    for (size_t i = 0; i < size; i++)
        *c++ = *a++ + *b++;

    return C;
}

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

__attribute__((used))
word* OL_mul(olvm_t* this, word* arguments)
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

    word* C = create_new_matrix(this, m, n, &A, &B);

    float* a = (float*) (ref(A, 3) + W);
    float* b = (float*) (ref(B, 3) + W);
    float* c = (float*) (ref(C, 3) + W);

    size_t size = rawstream_size(ref(A, 3)) / sizeof(fp_t); // todo: assert for matrix B size
    for (size_t i = 0; i < size; i++)
        *c++ = *a++ * *b++;

    return C;
}

__attribute__((used))
word* OL_T(olvm_t* this, word* arguments) // transpose
{
    word* A = (word*)car(arguments); arguments = (word*)cdr(arguments); // matrix A
    assert ((word)arguments == INULL);
    // todo: assert for matrix sizes

    size_t m = value(ref(A, 1));
    size_t n = value(ref(A, 2));

    word* B = create_new_matrix(this, n, m, &A);

    float* a = (float*) (ref(A, 3) + W);
    float* b = (float*) (ref(B, 3) + W);

    for (size_t i = 0; i < m; i++)
        for (size_t j = 0; j < n; j++)
            b[j*m + i] = a[i*n + j];

    return B;
}


// OLD: ==================================================
// typedef struct _matrix_t
// {
//     size_t rows;
//     size_t columns;
//     float* floats;
// } matrix_t;

// // create new float matrix m*n
// matrix_t* new(size_t m, size_t n)
// {
//     matrix_t* matrix = (matrix_t*)malloc(sizeof(matrix_t));
//     matrix->rows = m;
//     matrix->columns = n;
//     matrix->floats = (float*)malloc(m * n * sizeof(float));
// }

// void ann_random(matrix_t* matrix)
// {
//     float* x = matrix->floats;
//     size_t m = matrix->rows;
//     size_t n = matrix->columns;
//     for (size_t i = 0; i < m * n; i++)
//         x[i] = (2.0 * (float)rand() / (float)RAND_MAX) - 1.0;
// }

// void ann_read(matrix_t* matrix, char* filename)
// {
//     FILE* f = fopen(filename, "rb");
//     size_t todo = matrix->rows * matrix->columns;
//     size_t read;
//     if (f != NULL) {
//         read = fread(matrix->floats, sizeof(float), todo, f);
//         fclose(f);
//     }
// }

// void ann_write(matrix_t* matrix, char* filename)
// {
//     FILE* f = fopen(filename, "wb");
//     size_t todo = matrix->rows * matrix->columns;
//     if (f != NULL) {
//         fwrite(matrix->floats, sizeof(float), todo, f);
//         fclose(f);
//     }
// }

// void dot(matrix_t* A /* m*n */, matrix_t* B /* n*q */, matrix_t* C /* m*q */)
// {
//     size_t m = A->rows;
//     size_t n = A->columns;
//     size_t q = B->columns;

//     assert(B->rows == n);
//     assert(C->rows == m);
//     assert(C->columns == q);

//     float* a = A->floats;
//     float* b = B->floats;
//     float* c = C->floats;

//     for (size_t i = 0; i < m; i++) {
//         for (size_t j = 0; j < q; j++) {
//             c[i + j*m] = 0;
//             for(size_t k = 0; k < n; k++)
//                 c[i + j*m] += a[i + k*m] * b[k + j*n];
//         }
//     }
// }

// float at(matrix_t* matrix, size_t i, size_t j) {
//     assert(i <= matrix->rows);
//     assert(j <= matrix->columns);
//     return matrix->floats[i + j*matrix->rows];
// }

// float at1(matrix_t* matrix, size_t i, size_t j) {
//     assert(i <= matrix->rows);
//     assert(j <= matrix->columns);
//     return matrix->floats[(i-1) + (j-1)*matrix->rows];
// }

// // transpose
// void T(matrix_t* A, matrix_t* B)
// {
//     assert(A->rows == B->rows);
//     assert(A->columns == B->columns);
//     size_t m = A->rows;
//     size_t n = A->columns;
//     float* a = A->floats;
//     float* b = B->floats;

//     for (size_t i = 0; i < m; i++) {
//         for (size_t j = 0; j < n; j++) {
//             b[i + j*n] = a[j + i*n];
//         }
//     }
// }

// void add(matrix_t* A, matrix_t* B, matrix_t* C)
// {
//     assert(A->rows == B->rows);
//     assert(A->columns == B->columns);
//     assert(A->rows == C->rows);
//     assert(A->columns == C->columns);
//     size_t m = A->rows;
//     size_t n = A->columns;
//     float* a = A->floats;
//     float* b = B->floats;
//     float* c = C->floats;

//     for (size_t i = 0; i < m; i++) {
//         for (size_t j = 0; j < n; j++) {
//             c[i + j*n] = a[i + j*n] + b[i + j*n];
//         }
//     }
// }

// void mul(matrix_t* A, matrix_t* B, matrix_t* C)
// {
//     assert(A->rows == B->rows);
//     assert(A->columns == B->columns);
//     assert(A->rows == C->rows);
//     assert(A->columns == C->columns);
//     size_t m = A->rows;
//     size_t n = A->columns;
//     float* a = A->floats;
//     float* b = B->floats;
//     float* c = C->floats;

//     for (size_t i = 0; i < m; i++) {
//         for (size_t j = 0; j < n; j++) {
//             c[i + j*n] = a[i + j*n] * b[i + j*n];
//         }
//     }
// }

// void rsub1(matrix_t* A, matrix_t* B)
// {
//     assert(A->rows == B->rows);
//     assert(A->columns == B->columns);
//     size_t m = A->rows;
//     size_t n = A->columns;
//     float* a = A->floats;
//     float* b = B->floats;

//     for (size_t i = 0; i < m; i++) {
//         for (size_t j = 0; j < n; j++) {
//             b[i + j*n] = 1.0f - a[i + j*n];
//         }
//     }
// }

// void sigmoid(matrix_t* A, matrix_t* B)
// {
//     assert(A->rows == B->rows);
//     assert(A->columns == B->columns);
//     size_t m = A->rows;
//     size_t n = A->columns;
//     float* a = A->floats;
//     float* b = B->floats;

//     for (size_t i = 0; i < m; i++) {
//         for (size_t j = 0; j < n; j++) {
//             float x = a[i + j*n];
//             b[i + j*n] = 1.0f / (1.0f + exp(0.0f - x));
//         }
//     }
// }

// void sigmoid_d(matrix_t* A, matrix_t* B)
// {
//     assert(A->rows == B->rows);
//     assert(A->columns == B->columns);
//     size_t m = A->rows;
//     size_t n = A->columns;
//     float* a = A->floats;
//     float* b = B->floats;

//     for (size_t i = 0; i < m; i++) {
//         for (size_t j = 0; j < n; j++) {
//             float x = a[i + j*n];
//             b[i + j*n] = (x * (1.0f - x));
//         }
//     }
// }

// void ann_set(matrix_t*A, float* f)
// {
//     size_t m = A->rows;
//     size_t n = A->columns;
//     float* a = A->floats;
//     size_t i = m*n;
//     while (i--)
//         *a++ = *f++;
// }

// /*
// ;; (define ann (load-dynamic-library "libann.so"))
// ;; (define matrix_t* fft-void*)
// ;; (define size_t fft-int)
// ;; (define void fft-void)
// ;; (define new (ann matrix_t* "new" size_t size_t))
// ;; (define ann-random (ann void "ann_random" matrix_t*))
// ;; (define ann-dot (ann void "dot" matrix_t* matrix_t* matrix_t*))
// ;; (define ann-read (ann void "ann_read" matrix_t* type-string))
// ;; (define ann-write (ann void "ann_write" matrix_t* type-string))
// ;; (define ann-set (ann void "ann_set" matrix_t* (fft* fft-float)))
// ;; (define ann-sigmoid (ann void "sigmoid" matrix_t* matrix_t*))
// ;; (define ann-at (ann fft-float "at" matrix_t* size_t size_t))
// */