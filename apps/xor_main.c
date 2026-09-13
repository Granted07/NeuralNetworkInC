//
// Created by anjis on 12-09-2026.
//
#include <stdio.h>

#include "activation.h"
#include "loss.h"
#include "network.h"

int main() {

    double input_vals[4][2] = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };
    double target_vals[4] = {0, 1, 1, 0};

    Matrix *inputs[4];
    Matrix *targets[4];
    for (int i = 0; i < 4; i++) {
        inputs[i] = create_matrix(2, 1);
        matrix_set(inputs[i], 0, 0, input_vals[i][0]);
        matrix_set(inputs[i], 1, 0, input_vals[i][1]);

        targets[i] = create_matrix(1, 1);
        matrix_set(targets[i], 0, 0, target_vals[i]);
    }


    const int sizes[] = {2, 4, 1};
    double (*activations[])(double) = {sigmoid, sigmoid};
    double (*activation_derivatives[])(double) = {
        sigmoid_derivative,
        sigmoid_derivative
    };

    Network *network = network_create(
        sizes,
        2,
        activations,
        activation_derivatives
    );

    double learning_rate = 0.5;
    int epochs = 6000;

    for (int epoch = 0; epoch < epochs; epoch++) {
        double total_loss = 0.0;

        for (int i = 0; i < 4; i++) {
            Matrix *output = network_forward(network, inputs[i]);
            total_loss += mse_loss(output, targets[i]);
            network_backward(network, targets[i], learning_rate);
        }
        if (epoch % 2000 == 0) {
            printf("epoch %d, loss = %f\n", epoch, total_loss / 4.0);
        }
    }
    for (int i = 0; i < 4; i++) {
        Matrix *output = network_forward(network, inputs[i]);   // borrowed pointer
        printf("input: [%.0f, %.0f]  expected: %.0f  predicted: %.4f\n",
               input_vals[i][0], input_vals[i][1], target_vals[i],
               matrix_get(output, 0, 0));
    }



    for (int i = 0; i < 4; i++) {
        free_matrix(inputs[i]);
        free_matrix(targets[i]);
    }
    network_free(network);
    return 0;
}