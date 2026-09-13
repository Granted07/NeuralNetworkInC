//
// Created by anjis on 13-09-2026.
//

#include "../include/loss.h"

#include <assert.h>
#include <math.h>
#include <stdlib.h>

double mse_loss(const Matrix *a, const Matrix *y) {
    assert(a->cols == y->cols);
    assert(a->rows == y->rows);
    double sum = 0.0;
    for (int i = 0; i < a->rows; ++i) {
        for (int j = 0; j < a->cols; ++j) {
            sum += pow((a->matrix[i][j] - y->matrix[i][j]), 2);
        }
    }
    return (1.0/(a->rows*a->cols))*sum;
}

Matrix *mse_loss_derivative(const Matrix *a, const Matrix *y) {
    assert(a->cols == y->cols);
    assert(a->rows == y->rows);
    Matrix *m = create_matrix(a->rows, a->cols);
    for (int i = 0; i < m->rows; ++i) {
        for (int j = 0; j < m->cols; ++j) {
            m->matrix[i][j] = (2.0/(a->rows*a->cols))*(a->matrix[i][j] - y->matrix[i][j]);
        }
    }
    return m;
}
