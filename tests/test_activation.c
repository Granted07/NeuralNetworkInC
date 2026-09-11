//
// Created by anjis on 12-09-2026.
//
#include <stdio.h>
#include <math.h>
#include "../include/activation.h"

int approx_equal(double a, double b) {
    return fabs(a - b) < 1e-9;
}

int main(void) {
    // sigmoid(0) should be exactly 0.5
    printf("sigmoid(0) = %f, expected 0.5 -> %s\n",
           sigmoid(0), approx_equal(sigmoid(0), 0.5) ? "PASS" : "FAIL");

    // relu
    printf("relu(-3) = %f, expected 0 -> %s\n",
           relu(-3), approx_equal(relu(-3), 0) ? "PASS" : "FAIL");
    printf("relu(3) = %f, expected 3 -> %s\n",
           relu(3), approx_equal(relu(3), 3) ? "PASS" : "FAIL");

    // relu_derivative
    printf("relu_derivative(-3) = %f, expected 0 -> %s\n",
           relu_derivative(-3), approx_equal(relu_derivative(-3), 0) ? "PASS" : "FAIL");
    printf("relu_derivative(3) = %f, expected 1 -> %s\n",
           relu_derivative(3), approx_equal(relu_derivative(3), 1) ? "PASS" : "FAIL");

    // sigmoid_derivative(0) = sigmoid(0)*(1-sigmoid(0)) = 0.5*0.5 = 0.25
    printf("sigmoid_derivative(0) = %f, expected 0.25 -> %s\n",
           sigmoid_derivative(0), approx_equal(sigmoid_derivative(0), 0.25) ? "PASS" : "FAIL");

    return 0;
}