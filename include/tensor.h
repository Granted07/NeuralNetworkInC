//
// Created by anjis on 14-09-2026.
//

#ifndef NEURALNETWORKINC_TENSOR_H
#define NEURALNETWORKINC_TENSOR_H
#include <stdbool.h>

#include "matrix.h"

typedef struct Tensor{
    Matrix *data;
    Matrix *grad;
    bool requires_grad;

    struct Tensor *parent1;
    struct Tensor *parent2;
    void(*backward_fn)(struct Tensor *self);

    int ref_count;
}Tensor;

Tensor *tensor_create(Matrix *data, bool requires_grad);

Tensor *tensor_copy(Tensor *tensor); // how does copying a tensor effect the graph and parents
Tensor *tensor_add(Tensor *left, Tensor *right);
Tensor *tensor_sub(Tensor *left, Tensor *right);
Tensor *tensor_multi(Tensor *left, Tensor *right);
Tensor *tensor_hadamard(Tensor *left, Tensor *right);

Tensor* tensor_retain(Tensor *t);   // increment ref_count, return t (for convenient chaining)
void tensor_release(Tensor *t);     // decrement ref_count; if it hits 0, actually free it

void add_backward(Tensor *self);



#endif //NEURALNETWORKINC_TENSOR_H
