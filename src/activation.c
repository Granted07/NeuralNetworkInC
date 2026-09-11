//
// Created by anjis on 12-09-2026.
//

#include "../include/activation.h"
#include <math.h>

double sigmoid(const double x) {
    return (1.0 / (1.0 + exp(-x)));
}
double sigmoid_derivative(const double x) {
    return (sigmoid(x)*(1.0 - sigmoid(x)));
}
double relu(const double x) {
    return x > 0 ? x : 0;
}
double relu_derivative(const double x) {
    return x > 0 ? 1 : 0;
}