//
// test_loss.c — manual tests for loss.c
//

#include <stdio.h>
#include <math.h>
#include "../include/matrix.h"
#include "../include/loss.h"

static int  approx_equal(double a, double b) {
    return fabs(a - b) < 1e-9;
}

// a = [1, 2, 3]^T, y = [1, 0, 0]^T
// diffs: 0, 2, 3 -> squared: 0, 4, 9 -> sum = 13 -> mse = 13/3
static void test_mse_loss() {
    printf("--- test_mse_loss ---\n");

    Matrix *a = create_matrix(3, 1);
    matrix_set(a, 0, 0, 1);
    matrix_set(a, 1, 0, 2);
    matrix_set(a, 2, 0, 3);

    Matrix *y = create_matrix(3, 1);
    matrix_set(y, 0, 0, 1);
    matrix_set(y, 1, 0, 0);
    matrix_set(y, 2, 0, 0);

    double loss = mse_loss(a, y);
    double expected = 13.0 / 3.0;

    printf("loss = %.10f, expected = %.10f\n", loss, expected);
    printf(approx_equal(loss, expected) ? "PASS\n\n" : "FAIL\n\n");

    free_matrix(a);
    free_matrix(y);
}

// Same a, y as above.
// derivative_i = (2/3) * (a_i - y_i)
// = (2/3)*0, (2/3)*2, (2/3)*3 = 0, 4/3, 2
void test_mse_loss_derivative() {
    printf("--- test_mse_loss_derivative ---\n");

    Matrix *a = create_matrix(3, 1);
    matrix_set(a, 0, 0, 1);
    matrix_set(a, 1, 0, 2);
    matrix_set(a, 2, 0, 3);

    Matrix *y = create_matrix(3, 1);
    matrix_set(y, 0, 0, 1);
    matrix_set(y, 1, 0, 0);
    matrix_set(y, 2, 0, 0);

    Matrix *d = mse_loss_derivative(a, y);
    matrix_print(d);

    double e0 = 0.0;
    double e1 = 4.0 / 3.0;
    double e2 = 2.0;

    int pass = approx_equal(matrix_get(d, 0, 0), e0) &&
               approx_equal(matrix_get(d, 1, 0), e1) &&
               approx_equal(matrix_get(d, 2, 0), e2);

    printf(pass ? "PASS\n\n" : "FAIL\n\n");

    free_matrix(a);
    free_matrix(y);
    free_matrix(d);
}

int main(void) {
    test_mse_loss();
    test_mse_loss_derivative();
    return 0;
}