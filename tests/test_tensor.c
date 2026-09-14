//
// test_tensor.c — manual tests for tensor.c
//

#include <stdio.h>
#include <math.h>
#include "../include/matrix.h"
#include "../include/tensor.h"

int approx_equal(double a, double b) {
    return fabs(a - b) < 1e-9;
}

// Test 1: forward math for tensor_add
// a = [1, 2], b = [3, 4] -> c = a + b = [4, 6]
void test_add_forward() {
    printf("--- test_add_forward ---\n");

    Matrix *a_data = create_matrix(2, 1);
    matrix_set(a_data, 0, 0, 1);
    matrix_set(a_data, 1, 0, 2);
    Tensor *a = tensor_create(a_data, true);

    Matrix *b_data = create_matrix(2, 1);
    matrix_set(b_data, 0, 0, 3);
    matrix_set(b_data, 1, 0, 4);
    Tensor *b = tensor_create(b_data, true);

    Tensor *c = tensor_add(a, b);

    printf("c->data:\n");
    matrix_print(c->data);

    int pass = approx_equal(matrix_get(c->data, 0, 0), 4) &&
               approx_equal(matrix_get(c->data, 1, 0), 6);
    printf(pass ? "PASS\n\n" : "FAIL\n\n");

    tensor_release(c);
    tensor_release(a);
    tensor_release(b);
}

// Test 2: backward rule accumulates gradient into both parents correctly
// c = a + b. Manually seed c->grad (as if it came from further downstream),
// then call c's backward_fn directly and check a->grad / b->grad.
void test_add_backward() {
    printf("--- test_add_backward ---\n");

    Matrix *a_data = create_matrix(2, 1);
    matrix_set(a_data, 0, 0, 1);
    matrix_set(a_data, 1, 0, 2);
    Tensor *a = tensor_create(a_data, true);

    Matrix *b_data = create_matrix(2, 1);
    matrix_set(b_data, 0, 0, 3);
    matrix_set(b_data, 1, 0, 4);
    Tensor *b = tensor_create(b_data, true);

    Tensor *c = tensor_add(a, b);

    // Seed c's incoming gradient, as if some loss further downstream
    // produced this signal. Chosen arbitrarily: [10, 20].
    matrix_set(c->grad, 0, 0, 10);
    matrix_set(c->grad, 1, 0, 20);

    c->backward_fn(c);

    printf("a->grad (expect [10, 20]):\n");
    matrix_print(a->grad);
    printf("b->grad (expect [10, 20]):\n");
    matrix_print(b->grad);

    int pass = approx_equal(matrix_get(a->grad, 0, 0), 10) &&
               approx_equal(matrix_get(a->grad, 1, 0), 20) &&
               approx_equal(matrix_get(b->grad, 0, 0), 10) &&
               approx_equal(matrix_get(b->grad, 1, 0), 20);
    printf(pass ? "PASS\n\n" : "FAIL\n\n");

    tensor_release(c);
    tensor_release(a);
    tensor_release(b);
}

// Test 3: gradient accumulation — if a tensor is used twice (branching),
// its gradient should be the SUM of both contributions, not overwritten.
// c1 = a + b, c2 = a + b (two separate sums reusing a and b)
// Seed both c1->grad and c2->grad, run both backward_fns,
// a->grad should equal the sum of both contributions.
void test_add_backward_accumulates() {
    printf("--- test_add_backward_accumulates ---\n");

    Matrix *a_data = create_matrix(1, 1);
    matrix_set(a_data, 0, 0, 1);
    Tensor *a = tensor_create(a_data, true);

    Matrix *b_data = create_matrix(1, 1);
    matrix_set(b_data, 0, 0, 2);
    Tensor *b = tensor_create(b_data, true);

    Tensor *c1 = tensor_add(a, b);
    Tensor *c2 = tensor_add(a, b);

    matrix_set(c1->grad, 0, 0, 5);
    matrix_set(c2->grad, 0, 0, 7);

    c1->backward_fn(c1);
    c2->backward_fn(c2);

    printf("a->grad (expect 12 = 5 + 7):\n");
    matrix_print(a->grad);

    int pass = approx_equal(matrix_get(a->grad, 0, 0), 12);
    printf(pass ? "PASS\n\n" : "FAIL\n\n");

    tensor_release(c1);
    tensor_release(c2);
    tensor_release(a);
    tensor_release(b);
}

// Test 4: ref counting — 'a' is used by two tensors (b's parent AND c's parent).
// Releasing one should NOT free 'a' while the other still references it.
void test_ref_counting() {
    printf("--- test_ref_counting ---\n");

    Matrix *a_data = create_matrix(1, 1);
    matrix_set(a_data, 0, 0, 5);
    Tensor *a = tensor_create(a_data, true);   // ref_count = 1

    Matrix *b_data = create_matrix(1, 1);
    matrix_set(b_data, 0, 0, 1);
    Tensor *b = tensor_create(b_data, true);

    Tensor *c1 = tensor_add(a, b);   // retains a -> ref_count = 2
    Tensor *c2 = tensor_add(a, b);   // retains a again -> ref_count = 3

    printf("a->ref_count after two uses (expect 3): %d\n", a->ref_count);
    int pass = (a->ref_count == 3);

    tensor_release(c1);   // releases its claim on a -> ref_count = 2 (a should still be alive)
    printf("a->ref_count after releasing c1 (expect 2): %d\n", a->ref_count);
    pass = pass && (a->ref_count == 2);

    // 'a' should still be safely readable here
    printf("a->data still readable (expect 5): %.1f\n", matrix_get(a->data, 0, 0));
    pass = pass && approx_equal(matrix_get(a->data, 0, 0), 5);

    printf(pass ? "PASS\n\n" : "FAIL\n\n");

    tensor_release(c2);   // releases its claim on a and b -> a, b freed via cascade
    tensor_release(a);    // release our own original reference
    tensor_release(b);
}

int main(void) {
    test_add_forward();
    test_add_backward();
    test_add_backward_accumulates();
    test_ref_counting();
    return 0;
}