#include <ann.h>

#include <stdio.h>
#include <string.h>

__attribute__((used))
word OL_mwrite(olvm_t* this, word* arguments)
{
    word M = car(arguments); arguments = (word*)cdr(arguments); // matrix
    word F = car(arguments); arguments = (word*)cdr(arguments); // filename
    assert ((word)arguments == INULL);

    size_t flen = rawstream_size(F);
    char* filename = __builtin_alloca(flen + 1);
    memcpy(filename, &car(F), flen);
    filename[flen] = 0;

    FILE* file = fopen(filename, "wb");
    if (!file)
        return IFALSE;

    // заголовок (1 в конце значит что в файле флоаты)
    int32_t magic = *(int32_t*)"ann\x1";
    fwrite(&magic, sizeof(magic), 1, file);

    // размерность
    uint32_t m = matrix_m(M);
    uint32_t n = matrix_n(M);

    fwrite(&m, sizeof(m), 1, file);
    fwrite(&n, sizeof(n), 1, file);

    // матрица
    fp_t* f = matrix_floats(M);
    fwrite(f, sizeof(fp_t), m*n, file);

    fclose(file);
    return ITRUE;
}

__attribute__((used))
word OL_mread(olvm_t* this, word* arguments)
{
    word A = car(arguments); arguments = (word*)cdr(arguments); // filename
    assert ((word)arguments == INULL);

    size_t flen = rawstream_size(A);
    char* filename = __builtin_alloca(flen + 1);
    memcpy(filename, &car(A), flen);
    filename[flen] = 0;

    FILE* file = fopen(filename, "rb");
    if (!file)
        return IFALSE;

    // заголовок (1 в конце значит, что там флоаты)
    int32_t magic = 0; // = *(int32_t*)"ann\x1";
    size_t read;

    read = fread(&magic, sizeof(magic), 1, file);
    if (magic != *(int32_t*)"ann\x1")
        goto fail;

    // размерность
    uint32_t m = 0; // matrix_m(M)
    uint32_t n = 0; // matrix_n(M)

    read = fread(&m, sizeof(m), 1, file);
    read = fread(&n, sizeof(n), 1, file);

    if (m == 0 || n == 0)
        goto fail;

    // матрица
    word matrix = new_matrix(this, m, n);

    read = fread(matrix_floats(matrix), sizeof(fp_t), m*n, file);

    fclose(file);
    return matrix;
fail:
    fclose(file);
    return IFALSE;
}

__attribute__((used))
word OL_mreadE(olvm_t* this, word* arguments)
{
    word M = car(arguments); arguments = (word*)cdr(arguments); // matrix
    word F = car(arguments); arguments = (word*)cdr(arguments); // filename
    assert ((word)arguments == INULL);

    size_t flen = rawstream_size(F);
    char* filename = __builtin_alloca(flen + 1);
    memcpy(filename, &car(F), flen);
    filename[flen] = 0;

    FILE* file = fopen(filename, "rb");
    if (!file)
        return IFALSE;

    // заголовок (1 в конце значит, что это флоаты)
    int32_t magic = 0; // = *(int32_t*)"ann\x1";
    size_t read;

    read = fread(&magic, sizeof(magic), 1, file);
    if (magic != *(int32_t*)"ann\x1")
        goto fail;

    // размерность
    uint32_t m = 0; // == value(ref(M, 1));
    uint32_t n = 0; // == value(ref(M, 2));

    read = fread(&m, sizeof(m), 1, file);
    read = fread(&n, sizeof(n), 1, file);

    if (m == 0 || n == 0)
        goto fail;
    if (m != matrix_m(M) || n != matrix_n(M))
        goto fail;

    // матрица
    word matrix = M;
    read = fread(matrix_floats(matrix), sizeof(fp_t), m*n, file);

    fclose(file);
    return matrix;
fail:
    fclose(file);
    return IFALSE;
}
