//
// Created by anjis on 12-09-2026.
//

#include <stdlib.h>
#include <stdio.h>
#include "../include/matrix.h"

#include <assert.h>

// Create and del
Matrix *create_matrix(const int rows, const int cols) {
    Matrix *m = (Matrix *)malloc(sizeof(Matrix));
    if (m == NULL) {
        printf("Error allocating memory\n");
        exit(-1);
    }
    m->rows = rows;
    m->cols = cols;
    m->matrix = (double **)calloc(rows, sizeof(double *));
    if (m->matrix == NULL) {
        printf("Error allocating memory\n");
        exit(-1);
    }
    for (int i = 0; i < rows; i++) {
        m->matrix[i] = (double *)calloc(cols, sizeof(double));
        if (m->matrix[i] == NULL) {
            printf("Error allocating memory\n");
            exit(-1);
        }
    }
    return m;
}

void free_matrix(Matrix *m) {
    for (int i = 0; i < m->rows; i++) {
        free(m->matrix[i]);
    }
    free(m->matrix);
    free(m);
}

// Accessing and setting values
double matrix_get(const Matrix *m, int i, int j) {
    return m->matrix[i][j];
}
void matrix_set(Matrix *m, int i, int j, double value) {
    m->matrix[i][j] = value;
}

//Core ops
Matrix* copy_matrix(const Matrix *m) {
    Matrix *m1 = create_matrix(m->rows, m->cols);
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            m1->matrix[i][j] = m->matrix[i][j];
        }
    }
    return m1;
}

Matrix* add_matrix(const Matrix *m1, const Matrix *m2) {
    Matrix *m = create_matrix(m1->rows, m1->cols);
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            m->matrix[i][j] = m1->matrix[i][j] + m2->matrix[i][j];
        }
    }
    return m;
}

Matrix* multi_matrix(const Matrix *m1, const Matrix *m2) {
    assert(m1->cols == m2->rows);
    Matrix *m = create_matrix(m1->rows, m2->cols);
    for (int i = 0; i < m1->rows; i++) {
        for (int j = 0; j < m2->cols; j++) {
            m->matrix[i][j] = 0;
            for (int k = 0; k < m1->cols; k++) {
                m->matrix[i][j] += m1->matrix[i][k] * m2->matrix[k][j];
            }
        }
    }
    return m;
}

Matrix* transpose_matrix(const Matrix *m) {
    Matrix *m1 = create_matrix(m->cols, m->rows);
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            m1->matrix[j][i] = m->matrix[i][j];
        }
    }
    return m1;
}

// Function application
Matrix* matrix_apply(const Matrix *m, double (*func)(double)) {
    Matrix *result = create_matrix(m->rows, m->cols);
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            result->matrix[i][j] = func(m->matrix[i][j]);
        }
    }
    return result;
}

//Debug and utility
void matrix_print(const Matrix *m) {
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            printf("%.2f ", m->matrix[i][j]);
        }
        printf("\n");
    }
}

void matrix_fill_random(Matrix *m, const double min, const double max) {
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            m->matrix[i][j] = min + ((double)rand()/RAND_MAX) * (max-min);
        }
    }
}

// Hadamard product
Matrix* hadamard_matrix(const Matrix *m1, const Matrix *m2) {
    assert(m1->cols == m2->cols);
    assert(m1->rows == m2->rows);
    Matrix *m = create_matrix(m1->rows, m1->cols);
    for (int i = 0; i < m1->rows; i++) {
        for (int j = 0; j < m1->cols; j++) {
            m->matrix[i][j] = m1->matrix[i][j] * m2->matrix[i][j];
        }
    }
    return m;
}

Matrix *scale_matrix(const Matrix *m, double scale) {
    Matrix *m1 = create_matrix(m->rows, m->cols);
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            m1->matrix[i][j] = scale * m->matrix[i][j];
        }
    }
    return m1;
}

Matrix *subtract_matrix(const Matrix *m1, const Matrix *m2) {
    Matrix *m = create_matrix(m1->rows, m1->cols);
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            m->matrix[i][j] = m1->matrix[i][j] - m2->matrix[i][j];
        }
    }
    return m;
}