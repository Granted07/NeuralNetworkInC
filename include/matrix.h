//
// Created by anjis on 12-09-2026.
//

#ifndef NEURALNETWORKINC_MATRIX_H
#define NEURALNETWORKINC_MATRIX_H

typedef struct {
    int rows;
    int cols;
    double **matrix;
} Matrix;

// Create and del
Matrix* create_matrix(int rows, int cols);
void free_matrix(Matrix *m);

// Accessing and setting value
double matrix_get(const Matrix *m, int i, int j);
void matrix_set(Matrix *m, int i, int j, double value);

// Core ops
Matrix *copy_matrix(const Matrix *m);
Matrix* add_matrix(const Matrix *m1, const Matrix *m2);
Matrix* multi_matrix(const Matrix *m1, const Matrix *m2);
Matrix* transpose_matrix(const Matrix *m);

// Applying function elementwise by utilising a function pointer
Matrix* matrix_apply(const Matrix *m, double (*func)(double));

//Debug and utility
void matrix_print(const Matrix *m);
void matrix_fill_random(Matrix *m, double min, double max);

#endif //NEURALNETWORKINC_MATRIX_H
