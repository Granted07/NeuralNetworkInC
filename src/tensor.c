//
// Created by anjis on 14-09-2026.
//

#include "../include/tensor.h"

#include <stdio.h>
#include <stdlib.h>

void add_backward(Tensor *self) {
    for (int i = 0; i < self->grad->rows; ++i) {
        for (int j = 0; j < self->grad->cols; ++j) {
            self->parent1->grad->matrix[i][j] += self->grad->matrix[i][j];
            self->parent2->grad->matrix[i][j] += self->grad->matrix[i][j];
        }
    }
}

Tensor *tensor_create(Matrix *data, const bool requires_grad) {
    Tensor *tensor = malloc(sizeof(Tensor));
    if (!tensor) {
        fprintf(stderr, "tensor allocation failed\n");
        exit(-1);
    }

    tensor->grad = create_matrix(data->rows, data->cols);

    tensor->data = data;
    tensor->requires_grad = requires_grad;
    tensor->parent1 = NULL;
    tensor->parent2 = NULL;
    tensor->backward_fn = NULL;
    tensor->ref_count = 1;
    return tensor;
}

Tensor *tensor_retain(Tensor *t) {
    t->ref_count++;
    return t;
}

void tensor_release(Tensor *t) {
    if (!t) return;
    if (--t->ref_count == 0) {
        free_matrix(t->data);
        free_matrix(t->grad);
        tensor_release(t->parent1);
        tensor_release(t->parent2);
        free(t);
    }
}

Tensor *tensor_add(Tensor *left, Tensor *right) {
    Matrix *result_data = add_matrix(left->data, right->data);
    Tensor *result = tensor_create(result_data, (left->requires_grad || right->requires_grad));

    result->parent1 = tensor_retain(left);
    result->parent2 = tensor_retain(right);
    result->backward_fn = add_backward;

    return result;
}