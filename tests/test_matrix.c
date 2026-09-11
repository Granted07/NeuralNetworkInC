//
// Created by anjis on 12-09-2026.
//
//
// test_matrix.c — manual tests for matrix.c
//

#include <stdio.h>
#include <math.h>
#include "../include/matrix.h"

// A simple activation function to test matrix_apply with
static double square(double x) {
    return x * x;
}

// Helper to compare doubles with a small tolerance
static int approx_equal(double a, double b) {
    return fabs(a - b) < 1e-9;
}

static void test_add() {
    printf("--- test_add ---\n");
    Matrix *a = create_matrix(2, 2);
    Matrix *b = create_matrix(2, 2);

    matrix_set(a, 0, 0, 1); matrix_set(a, 0, 1, 2);
    matrix_set(a, 1, 0, 3); matrix_set(a, 1, 1, 4);

    matrix_set(b, 0, 0, 5); matrix_set(b, 0, 1, 6);
    matrix_set(b, 1, 0, 7); matrix_set(b, 1, 1, 8);

    Matrix *result = add_matrix(a, b);
    matrix_print(result);

    // Expected: [[6, 8], [10, 12]]
    if (approx_equal(matrix_get(result, 0, 0), 6) &&
        approx_equal(matrix_get(result, 0, 1), 8) &&
        approx_equal(matrix_get(result, 1, 0), 10) &&
        approx_equal(matrix_get(result, 1, 1), 12)) {
        printf("PASS\n\n");
    } else {
        printf("FAIL\n\n");
    }

    free_matrix(a);
    free_matrix(b);
    free_matrix(result);
}

static void test_multiply() {
    printf("--- test_multiply ---\n");
    // A is 2x3, B is 3x1 — hand-computed example
    Matrix *a = create_matrix(2, 3);
    Matrix *b = create_matrix(3, 1);

    matrix_set(a, 0, 0, 1); matrix_set(a, 0, 1, 2); matrix_set(a, 0, 2, 3);
    matrix_set(a, 1, 0, 4); matrix_set(a, 1, 1, 5); matrix_set(a, 1, 2, 6);

    matrix_set(b, 0, 0, 1);
    matrix_set(b, 1, 0, 2);
    matrix_set(b, 2, 0, 3);

    // Expected:
    // row0: 1*1 + 2*2 + 3*3 = 14
    // row1: 4*1 + 5*2 + 6*3 = 32
    Matrix *result = multi_matrix(a, b);
    matrix_print(result);

    if (approx_equal(matrix_get(result, 0, 0), 14) &&
        approx_equal(matrix_get(result, 1, 0), 32)) {
        printf("PASS\n\n");
    } else {
        printf("FAIL\n\n");
    }

    free_matrix(a);
    free_matrix(b);
    free_matrix(result);
}

static void test_transpose() {
    printf("--- test_transpose ---\n");
    // A is 2x3
    Matrix *a = create_matrix(2, 3);
    matrix_set(a, 0, 0, 1); matrix_set(a, 0, 1, 2); matrix_set(a, 0, 2, 3);
    matrix_set(a, 1, 0, 4); matrix_set(a, 1, 1, 5); matrix_set(a, 1, 2, 6);

    Matrix *result = transpose_matrix(a);
    matrix_print(result);

    // Expected: 3x2 -> [[1,4],[2,5],[3,6]]
    if (result->rows == 3 && result->cols == 2 &&
        approx_equal(matrix_get(result, 0, 0), 1) &&
        approx_equal(matrix_get(result, 0, 1), 4) &&
        approx_equal(matrix_get(result, 1, 0), 2) &&
        approx_equal(matrix_get(result, 1, 1), 5) &&
        approx_equal(matrix_get(result, 2, 0), 3) &&
        approx_equal(matrix_get(result, 2, 1), 6)) {
        printf("PASS\n\n");
    } else {
        printf("FAIL\n\n");
    }

    free_matrix(a);
    free_matrix(result);
}

static void test_apply() {
    printf("--- test_apply ---\n");
    Matrix *a = create_matrix(2, 2);
    matrix_set(a, 0, 0, 1); matrix_set(a, 0, 1, 2);
    matrix_set(a, 1, 0, 3); matrix_set(a, 1, 1, 4);

    Matrix *result = matrix_apply(a, square);
    matrix_print(result);

    // Expected: [[1, 4], [9, 16]]
    if (approx_equal(matrix_get(result, 0, 0), 1) &&
        approx_equal(matrix_get(result, 0, 1), 4) &&
        approx_equal(matrix_get(result, 1, 0), 9) &&
        approx_equal(matrix_get(result, 1, 1), 16)) {
        printf("PASS\n\n");
        } else {
            printf("FAIL\n\n");
        }

    free_matrix(a);
    free_matrix(result);
}

int main(void) {
    test_add();
    test_multiply();
    test_transpose();
    test_apply();
    return 0;
}