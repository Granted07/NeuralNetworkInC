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
    if (l->x) free(l->x);
    if (l->a) free(l->a);
    if (l->z) free(l->z);
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