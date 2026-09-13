//
// test_layer.c — manual tests for layer.c
//

#include <stdio.h>
#include <math.h>
#include "../include/matrix.h"
#include "../include/activation.h"
#include "../include/layer.h"

int approx_equal(double a, double b) {
    return fabs(a - b) < 1e-9;
}

// Test 1: sigmoid layer, weights/bias chosen so z = 0 exactly
// -> sigmoid(0) = 0.5 exactly, easy to verify by hand.
void test_layer_sigmoid() {
    printf("--- test_layer_sigmoid ---\n");

    // 2 inputs -> 2 outputs
    Layer *l = create_layer(2, 2, sigmoid, sigmoid_derivative);

    // W = [[1, 0], [0, 1]]  (identity)
    matrix_set(l->weights, 0, 0, 1); matrix_set(l->weights, 0, 1, 0);
    matrix_set(l->weights, 1, 0, 0); matrix_set(l->weights, 1, 1, 1);

    // x = [1, 2]^T  ->  Wx = [1, 2]^T
    Matrix *x = create_matrix(2, 1);
    matrix_set(x, 0, 0, 1);
    matrix_set(x, 1, 0, 2);

    // b = [-1, -2]^T  ->  z = Wx + b = [0, 0]^T
    matrix_set(l->bias, 0, 0, -1);
    matrix_set(l->bias, 1, 0, -2);

    Matrix *a = layer_forward(l, x);   // borrowed pointer — do NOT free this

    printf("output:\n");
    matrix_print(a);

    // Expected: sigmoid(0) = 0.5 for both entries
    if (approx_equal(matrix_get(a, 0, 0), 0.5) &&
        approx_equal(matrix_get(a, 1, 0), 0.5)) {
        printf("PASS\n\n");
    } else {
        printf("FAIL\n\n");
    }

    free_matrix(x);     // we own x, we free it — but only AFTER layer_forward
                          // is done using it internally for this test's purposes
    free_layer(l);       // frees l->weights, l->bias, l->x, l->z, l->a internally
}

// Test 2: relu layer, weights/bias chosen for an easy hand-computed result.
void test_layer_relu() {
    printf("--- test_layer_relu ---\n");

    // 3 inputs -> 2 outputs
    Layer *l = create_layer(3, 2, relu, relu_derivative);

    // W row0 = [1, 1, 1], row1 = [-1, -1, -1]
    matrix_set(l->weights, 0, 0, 1); matrix_set(l->weights, 0, 1, 1); matrix_set(l->weights, 0, 2, 1);
    matrix_set(l->weights, 1, 0, -1); matrix_set(l->weights, 1, 1, -1); matrix_set(l->weights, 1, 2, -1);

    // bias = [0, 0]
    matrix_set(l->bias, 0, 0, 0);
    matrix_set(l->bias, 1, 0, 0);

    // x = [1, 2, 3]^T
    Matrix *x = create_matrix(3, 1);
    matrix_set(x, 0, 0, 1);
    matrix_set(x, 1, 0, 2);
    matrix_set(x, 2, 0, 3);

    // Wx = [1+2+3, -(1+2+3)] = [6, -6]
    // z = Wx + b = [6, -6]
    // a = relu(z) = [6, 0]
    Matrix *a = layer_forward(l, x);

    printf("output:\n");
    matrix_print(a);

    if (approx_equal(matrix_get(a, 0, 0), 6) &&
        approx_equal(matrix_get(a, 1, 0), 0)) {
        printf("PASS\n\n");
    } else {
        printf("FAIL\n\n");
    }

    free_matrix(x);
    free_layer(l);
}

int main(void) {
    test_layer_sigmoid();
    test_layer_relu();
    return 0;
}