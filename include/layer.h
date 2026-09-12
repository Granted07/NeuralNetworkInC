//
// Created by anjis on 12-09-2026.
//

#ifndef NEURALNETWORKINC_LAYER_H
#define NEURALNETWORKINC_LAYER_H
#include "matrix.h"

typedef struct {
    int input_size;
    int output_size;

    Matrix *weights;
    Matrix *bias;

    Matrix *x; // input
    Matrix *a; // post activation output
    Matrix *z; // pre activation cache

    double (*activation)(double);
    double (*activation_derivative)(double);
} Layer;

Layer *create_layer(int input_size, int output_size, double (*activation)(double), double (*activation_derivative)(double));
void free_layer(Layer *layer);

Matrix *layer_forward(Layer *l, Matrix *x);


#endif //NEURALNETWORKINC_LAYER_H
