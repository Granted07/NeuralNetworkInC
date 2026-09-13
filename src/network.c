//
// Created by anjis on 13-09-2026.
//

#include "../include/network.h"

#include <stdio.h>
#include <stdlib.h>

#include "loss.h"

Network *network_create(
    const int *sizes,
    const int num_layers,
    double (**activations)(double),
    double(**activation_derivatives)(double)
    ) {
    Network *net = (Network *)malloc(sizeof(Network));
    if (!net) {
        printf("Malloc failed for network");
        exit(-1);
    }
    net->num_layers= num_layers;
    net->layers = (Layer **)calloc(num_layers, sizeof(Layer *));
    if (!net->layers) {
        printf("Malloc failed for layers");
        exit(-1);
    }
    for (int i=0; i<num_layers; i++) {
        net->layers[i] = create_layer(sizes[i], sizes[i+1], activations[i], activation_derivatives[i]);
    }
    return net;
}

Matrix *network_forward(Network *n, Matrix *x) {
    Matrix *current = x;
    for (int i=0; i<n->num_layers; i++) {
        current = layer_forward(n->layers[i], current);
    }
    return current;
}

void network_free(Network *n) {
    for (int i=0; i<n->num_layers; i++) {
        free(n->layers[i]);
    }
    free(n);
}

void network_backward_helper(Network *n, const int i, const Matrix *grad, const double learning_rate) {
    if (i < 0) return;

    Matrix *prev_grad = layer_backward(n->layers[i], grad, learning_rate);
    network_backward_helper(n, i - 1, prev_grad, learning_rate);
    free_matrix(prev_grad);
}

void network_backward(Network *n, const Matrix *y, double learning_rate) {
    const Matrix *activation = n->layers[n->num_layers-1]->a;
    Matrix *grad = mse_loss_derivative(activation, y);
    for (int i=n->num_layers-1; i>=0; i--) {
        Matrix *new_grad = layer_backward(n->layers[i], grad, learning_rate);
        free_matrix(grad);
        grad = new_grad;
    }
    free_matrix(grad);
}