//
// test_layer_backward.c — hand-verified test for layer_backward
//

#include <stdio.h>
#include <math.h>
#include "../include/matrix.h"
#include "../include/layer.h"

int approx_equal(double a, double b) {
    return fabs(a - b) < 1e-9;
}

// Linear activation: identity function and its derivative (constant 1)
double linear(double x) { return x; }
double linear_derivative(double x) { (void)x; return 1.0; }

// Setup:
//   Layer: 2 inputs -> 1 output, linear activation
//   W = [2, 3]        (1x2)
//   b = [1]           (1x1)
//   x = [1, 2]^T      (2x1)
//
// Forward pass (hand-computed):
//   z = Wx + b = (2*1 + 3*2) + 1 = 2 + 6 + 1 = 9
//   a = linear(z) = 9
//
// Suppose dL/da = [2] (pretend the loss gradient w.r.t. output is 2 -- a
// simple, arbitrary upstream value chosen to make arithmetic easy).
//
// Backward pass (hand-computed):
//   sigma'(z) = 1 (linear activation)
//   dL/dz = dL/da * sigma'(z) = 2 * 1 = 2
//
//   dL/dW = dL/dz * x^T = 2 * [1, 2] = [2, 4]      (1x2)
//   dL/db = dL/dz = [2]                             (1x1)
//   dL/dx = W^T * dL/dz = [2,3]^T * 2 = [4, 6]      (2x1)
//
// With learning_rate = 0.1:
//   new_W = W - 0.1*[2,4] = [2 - 0.2, 3 - 0.4] = [1.8, 2.6]
//   new_b = b - 0.1*[2]   = [1 - 0.2]           = [0.8]
void test_layer_backward() {
    printf("--- test_layer_backward ---\n");

    Layer *l = create_layer(2, 1, linear, linear_derivative);

    matrix_set(l->weights, 0, 0, 2);
    matrix_set(l->weights, 0, 1, 3);
    matrix_set(l->bias, 0, 0, 1);

    Matrix *x = create_matrix(2, 1);
    matrix_set(x, 0, 0, 1);
    matrix_set(x, 1, 0, 2);

    Matrix *a = layer_forward(l, x);   // borrowed pointer, not freed directly
    printf("forward output (expect 9.0):\n");
    matrix_print(a);

    Matrix *dL_da = create_matrix(1, 1);
    matrix_set(dL_da, 0, 0, 2);

    Matrix *dL_dx = layer_backward(l, dL_da, 0.1);

    printf("dL_dx (expect [4, 6]):\n");
    matrix_print(dL_dx);

    printf("updated weights (expect [1.8, 2.6]):\n");
    matrix_print(l->weights);

    printf("updated bias (expect [0.8]):\n");
    matrix_print(l->bias);

    int pass =
        approx_equal(matrix_get(dL_dx, 0, 0), 4.0) &&
        approx_equal(matrix_get(dL_dx, 1, 0), 6.0) &&
        approx_equal(matrix_get(l->weights, 0, 0), 1.8) &&
        approx_equal(matrix_get(l->weights, 0, 1), 2.6) &&
        approx_equal(matrix_get(l->bias, 0, 0), 0.8);

    printf(pass ? "PASS\n\n" : "FAIL\n\n");

    free_matrix(dL_da);
    free_matrix(dL_dx);
    free_matrix(x);
    free_layer(l);
}

int main(void) {
    test_layer_backward();
    return 0;
}