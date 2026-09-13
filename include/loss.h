//
// Created by anjis on 13-09-2026.
//

#ifndef NEURALNETWORKINC_LOSS_H
#define NEURALNETWORKINC_LOSS_H
#include "matrix.h"

double mse_loss(const Matrix *a, const Matrix *y);
Matrix *mse_loss_derivative(const Matrix *a, const Matrix *y);


#endif //NEURALNETWORKINC_LOSS_H
