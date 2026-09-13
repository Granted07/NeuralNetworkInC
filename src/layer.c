//
// Created by anjis on 12-09-2026.
//

#include "../include/layer.h"
#include "../include/matrix.h"
#include <stdio.h>
#include <stdlib.h>

Layer *create_layer(int input_size, int output_size, double (*activation)(double), double (*activation_derivative)(double)) {
    Layer *l = (Layer *) malloc(sizeof(Layer));
    if (!l) {
        printf("Error allocating memory for layer\n");
        exit(-1);
    }
    l->input_size = input_size;
    l->output_size = output_size;
    l->activation = activation;
    l->activation_derivative = activation_derivative;

    l->weights = create_matrix(output_size, input_size);
    l->bias = create_matrix(output_size, 1);
    matrix_fill_random(l->weights, -1.0, 1.0);
    matrix_fill_random(l->bias, -1.0, 1.0);

    l->x = NULL;
    l->a = NULL;
    l->z = NULL;
    return l;
}

void free_layer(Layer *l) {
    free_matrix(l->weights);
    free_matrix(l->bias);
    if (l->a) free_matrix(l->a);
    if (l->z) free_matrix(l->z);
    free(l);
}

// Runs the forward pass. Returns a pointer to the layer's internally
// cached output owned by layer. Do not free this pointer.
// It remains valid until the next call to layer_forward
// or until free_layer(l) is called. If the value of l.a is needed to outlive
// that, copy it explicitly with matrix_copy(l->a).
Matrix *layer_forward(Layer *l, Matrix *x) {
    Matrix *wx = multi_matrix(l->weights, x);
    Matrix *z = add_matrix(wx, l->bias);
    free_matrix(wx);
    Matrix *a = matrix_apply(z, l->activation);
    if (l->z) free_matrix(l->z);
    if (l->a) free_matrix(l->a);
    l->x = x;
    l->z = z;
    l->a = a;
    return a;
}

Matrix* layer_backward(Layer *l, const Matrix *dL_da, double learning_rate) {
    Matrix *sigma_prime_z = matrix_apply(l->z, l->activation_derivative);
    Matrix *dL_dz = hadamard_matrix(dL_da, sigma_prime_z);
    free_matrix(sigma_prime_z);
    Matrix *xt = transpose_matrix(l->x);
    Matrix *dL_dW = multi_matrix(dL_dz, xt);
    free_matrix(xt);
    Matrix *ndL_dW = scale_matrix(dL_dW, learning_rate);
    Matrix *ndL_dz = scale_matrix(dL_dz, learning_rate);
    free_matrix(dL_dW);
    Matrix *Wt = transpose_matrix(l->weights);
    Matrix *dL_dx = multi_matrix(Wt, dL_dz);
    free_matrix(Wt);
    free_matrix(dL_dz);
    Matrix *new_weights = subtract_matrix(l->weights, ndL_dW);
    free_matrix(l->weights);
    l->weights = new_weights;
    Matrix *new_biases = subtract_matrix(l->bias, ndL_dz);
    free_matrix(l->bias);
    l->bias = new_biases;
    free_matrix(ndL_dW);
    free_matrix(ndL_dz);
    return dL_dx;
}