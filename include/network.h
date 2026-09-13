//
// Created by anjis on 13-09-2026.
//

#ifndef NEURALNETWORKINC_NETWORK_H
#define NEURALNETWORKINC_NETWORK_H
#include "layer.h"


typedef struct {
    Layer **layers;
    int num_layers;
}Network;

Network *network_create(
    const int *sizes,
    int num_layers,
    double (**activations)(double),
    double(**activation_derivatives)(double)
    );
void network_free(Network *n);
Matrix *network_forward(Network *n, Matrix *x);
void network_backward_helper(Network *n, int i, const Matrix *grad, double learning_rate);
void network_backward(Network *n, const Matrix *y, double learning_rate);


#endif //NEURALNETWORKINC_NETWORK_H
