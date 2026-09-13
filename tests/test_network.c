//
// test_network.c — manual tests for network.c
//

#include <stdio.h>
#include <math.h>
#include "../include/matrix.h"
#include "../include/activation.h"
#include "../include/layer.h"
#include "../include/network.h"

int approx_equal(double a, double b) {
    return fabs(a - b) < 1e-6;
}

// Network: 2 inputs -> 2 (relu) -> 1 (sigmoid)
//
// Layer 0: W0 = identity, b0 = [0, 0], activation = relu
//   x = [3, -2]^T
//   z0 = W0*x + b0 = [3, -2]
//   a0 = relu(z0) = [3, 0]
//
// Layer 1: W1 = [1, 1], b1 = [0], activation = sigmoid
//   z1 = W1*a0 + b1 = 1*3 + 1*0 + 0 = 3
//   a1 = sigmoid(3) ≈ 0.9525741268
void test_network_forward() {
    printf("--- test_network_forward ---\n");

    int sizes[] = {2, 2, 1};
    double (*acts[])(double) = { relu, sigmoid };
    double (*act_derivs[])(double) = { relu_derivative, sigmoid_derivative };

    Network *net = network_create(sizes, 2, acts, act_derivs);

    // Layer 0: identity weights, zero bias
    Layer *l0 = net->layers[0];
    matrix_set(l0->weights, 0, 0, 1); matrix_set(l0->weights, 0, 1, 0);
    matrix_set(l0->weights, 1, 0, 0); matrix_set(l0->weights, 1, 1, 1);
    matrix_set(l0->bias, 0, 0, 0);
    matrix_set(l0->bias, 1, 0, 0);

    // Layer 1: weights [1, 1], zero bias
    Layer *l1 = net->layers[1];
    matrix_set(l1->weights, 0, 0, 1); matrix_set(l1->weights, 0, 1, 1);
    matrix_set(l1->bias, 0, 0, 0);

    Matrix *x = create_matrix(2, 1);
    matrix_set(x, 0, 0, 3);
    matrix_set(x, 1, 0, -2);

    Matrix *output = network_forward(net, x);   // borrowed pointer, do not free

    printf("output:\n");
    matrix_print(output);

    double expected = 1.0 / (1.0 + exp(-3.0));
    printf("expected: %.10f\n", expected);

    if (approx_equal(matrix_get(output, 0, 0), expected)) {
        printf("PASS\n\n");
    } else {
        printf("FAIL\n\n");
    }

    free_matrix(x);
    network_free(net);
}

int main(void) {
    test_network_forward();
    return 0;
}