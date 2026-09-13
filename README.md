# NeuralNetworkInC

A small, reusable feed-forward neural-network library written in C11.

The project implements the core building blocks needed to create and train fully connected neural networks:

- Dynamic matrices and matrix operations
- Configurable dense layers
- Forward propagation
- Backpropagation and gradient-descent updates
- Pluggable activation functions
- Mean squared error loss
- CMake-based builds
- Unit-style executable tests

The neural-network implementation is kept separate from applications. The reusable library is built as `nncore`, while programs in `apps/` link against it.

## Project structure

```text
.
├── CMakeLists.txt
├── include/
│   ├── activation.h    # Activation functions and derivatives
│   ├── layer.h        # Dense-layer API
│   ├── loss.h         # Loss-function API
│   ├── matrix.h       # Matrix type and operations
│   └── network.h      # Network API
├── src/
│   ├── activation.c
│   ├── layer.c
│   ├── loss.c
│   ├── matrix.c
│   └── network.c
├── apps/               # Executable applications using nncore
└── tests/              # Executable tests for individual components
```

## Design overview

The library uses a layered design:

```text
Application
    |
    v
Network
    |
    v
Layer
    |
    v
Matrix operations
```

### Matrix

`Matrix` is the fundamental data structure:

```c
typedef struct {
    int rows;
    int cols;
    double **matrix;
} Matrix;
```

Matrices are dynamically allocated and use row-major indexing through a two-dimensional pointer structure. The matrix API provides:

- Creation and destruction
- Element access and assignment
- Copying
- Addition and subtraction
- Matrix multiplication
- Transposition
- Scalar multiplication
- Hadamard, or element-wise, multiplication
- Element-wise function application
- Random initialization
- Printing for debugging

Matrix dimensions are checked with assertions for operations where compatible shapes are required.

### Layer

A `Layer` represents one fully connected neural-network layer. It stores:

- Input and output sizes
- Weight matrix
- Bias vector
- Cached input
- Cached pre-activation values
- Cached post-activation values
- Activation function
- Activation derivative

For a layer with `input_size` inputs and `output_size` neurons:

```text
W: output_size x input_size
b: output_size x 1
x: input_size x 1
```

The forward pass computes:

```text
z = W x + b
a = activation(z)
```

The layer caches `x`, `z`, and `a` because they are required during backpropagation.

### Network

A `Network` owns an ordered collection of layers:

```c
typedef struct {
    Layer **layers;
    int num_layers;
} Network;
```

The `sizes` array passed to `network_create` describes the dimensions of the complete network. For example:

```c
int sizes[] = {input_size, hidden_size, output_size};
```

This creates two trainable layers:

```text
input_size -> hidden_size -> output_size
```

The number of layers is `number_of_sizes - 1`.

## Forward propagation

`network_forward` passes an input matrix through every layer in sequence:

```text
x -> layer 0 -> layer 1 -> ... -> final output
```

Each layer applies its own weights, bias, and activation function. The returned output is owned by the final layer and must not be freed by the caller.

Example:

```c
Matrix *input = create_matrix(input_size, 1);
Matrix *output = network_forward(network, input);

matrix_print(output);

free_matrix(input);
network_free(network);
```

The output remains valid until the next forward pass through the final layer or until the network is freed. If it must be retained, copy it with `copy_matrix`.

## Backpropagation and training

The network currently uses mean squared error:

```text
MSE = 1/n * sum((prediction - target)^2)
```

Its derivative is:

```text
dMSE/da = 2/n * (prediction - target)
```

For each layer, backpropagation:

1. Computes the activation derivative at the cached pre-activation values.
2. Computes the local gradient:

   ```text
   dL/dz = dL/da * activation'(z)
   ```

3. Computes the weight gradient:

   ```text
   dL/dW = dL/dz * x^T
   ```

4. Computes the bias gradient:

   ```text
   dL/db = dL/dz
   ```

5. Propagates the gradient to the previous layer:

   ```text
   dL/dx = W^T * dL/dz
   ```

6. Updates weights and biases using gradient descent:

   ```text
   W = W - learning_rate * dL/dW
   b = b - learning_rate * dL/db
   ```

Training code generally follows this pattern:

```c
for (int epoch = 0; epoch < epoch_count; ++epoch) {
    Matrix *prediction = network_forward(network, input);

    double loss = mse_loss(prediction, target);
    network_backward(network, target, learning_rate);

    printf("epoch=%d loss=%f\n", epoch, loss);
}
```

`network_backward` must be called after `network_forward`, because backpropagation uses the values cached by the forward pass.

## Activation functions

Activation functions are represented by C function pointers:

```c
double (*activation)(double);
double (*activation_derivative)(double);
```

The project includes:

- `sigmoid`
- `sigmoid_derivative`
- `relu`
- `relu_derivative`

The activation function and its derivative are selected independently for each layer. This allows different layers to use different activations.

To add a custom activation, implement both the function and its derivative:

```c
double tanh_activation(double x) {
    return tanh(x);
}

double tanh_derivative(double x) {
    double value = tanh(x);
    return 1.0 - value * value;
}
```

Then pass arrays of function pointers to `network_create`:

```c
double (*activations[])(double) = {
    tanh_activation,
    sigmoid
};

double (*activation_derivatives[])(double) = {
    tanh_derivative,
    sigmoid_derivative
};
```

The derivative must use the same input convention as the layer implementation: it receives the pre-activation value `z`.

## Creating a network

The main constructor is:

```c
Network *network_create(
    const int *sizes,
    int num_layers,
    double (**activations)(double),
    double (**activation_derivatives)(double)
);
```

Example for a network with three dense layers:

```c
int sizes[] = {8, 16, 12, 3};

double (*activations[])(double) = {
    relu,
    relu,
    sigmoid
};

double (*activation_derivatives[])(double) = {
    relu_derivative,
    relu_derivative,
    sigmoid_derivative
};

Network *network = network_create(
    sizes,
    3,
    activations,
    activation_derivatives
);
```

The arrays must contain exactly `num_layers` entries. The `sizes` array must contain `num_layers + 1` dimensions.

## Memory ownership

The library uses explicit allocation and deallocation.

### Matrices

Every matrix returned by functions such as `create_matrix`, `copy_matrix`, `add_matrix`, and `multi_matrix` is heap-allocated and must eventually be released with:

```c
free_matrix(matrix);
```

### Layers

Layers created with `create_layer` should be released with:

```c
free_layer(layer);
```

### Networks

Networks created with `network_create` should be released with:

```c
network_free(network);
```

Freeing a network releases its layer storage and network storage. When working directly with layers, prefer `free_layer` so that all layer-owned matrices are released correctly.

## Building

Requirements:

- CMake 3.20 or newer
- A C11-compatible compiler
- A linker and math library

Configure and build from the project root:

```bash
cmake -S . -B build
cmake --build build
```

The project enables:

```text
-Wall -Wextra -Werror
```

The static library is produced as `nncore`.

## Running tests

Build the project, then run:

```bash
ctest --test-dir build --output-on-failure
```

The test programs cover:

- Matrix operations
- Activation functions
- Layer forward propagation
- Layer backpropagation
- Network forward propagation
- Loss functions

Individual test executables can also be run directly from the build directory.

## Reusing the library in another application

An application only needs to:

1. Include the required public headers.
2. Link against `nncore`.
3. Define network dimensions.
4. Select activation functions and derivatives.
5. Create input and target matrices.
6. Run forward propagation.
7. Run backpropagation when training.
8. Release allocated resources.

For CMake projects, the reusable target can be linked like this:

```cmake
add_executable(my_app my_app.c)
target_link_libraries(my_app PRIVATE nncore)
```

Because the public headers are exposed through the library target, consumers can include them directly:

```c
#include "activation.h"
#include "matrix.h"
#include "network.h"
```

## Scalability and extension points

The current architecture is reusable because network topology and behavior are data-driven rather than hard-coded.

### Change network topology without changing the engine

The same implementation supports different numbers of inputs, hidden layers, neurons, and outputs by changing the `sizes` array and activation arrays. The forward and backward algorithms iterate over `num_layers`, so they do not depend on a specific topology.

### Add activation functions

New activation functions can be added without modifying matrix, layer, or network code. The function-pointer interface makes activation behavior configurable per layer.

### Add loss functions

Loss functions are isolated in `loss.c` and `loss.h`. A different loss can be added by implementing:

```c
double custom_loss(const Matrix *prediction, const Matrix *target);
Matrix *custom_loss_derivative(
    const Matrix *prediction,
    const Matrix *target
);
```

The network training entry point can then be extended to accept loss callbacks instead of being tied to MSE.

### Add optimizers

The current update rule is basic gradient descent. The layer structure provides a natural place to add optimizer state such as:

- Momentum velocities
- Adam first and second moments
- Per-parameter learning rates
- Weight decay

Optimizer state should be stored alongside each layer's weights and biases, and the update logic should be separated from gradient calculation.

### Add batching

The current implementation operates on column vectors representing one sample at a time. Mini-batch support can be added by representing multiple samples as matrix columns and updating:

- Bias broadcasting
- Activation application
- Loss reduction
- Gradient averaging
- Input and output shape validation

### Improve initialization

Weights and biases currently use uniform random values in `[-1, 1]`. Larger networks can benefit from initialization strategies such as:

- Xavier/Glorot initialization
- He initialization
- Zero or controlled bias initialization

These strategies can be added to the layer-construction path without changing the network interface.

### Improve performance

The matrix implementation is intentionally simple and educational. For larger networks, performance can be improved by:

- Contiguous matrix storage
- Fewer temporary allocations
- In-place matrix operations
- BLAS-backed matrix multiplication
- SIMD or multithreaded kernels
- Better cache-aware layouts

These optimizations can be implemented behind the matrix API while preserving the layer and network interfaces.

## Current scope and limitations

This project currently focuses on a simple dense neural-network core. It does not yet provide:

- Convolutional or recurrent layers
- Mini-batch training
- Optimizers beyond basic gradient descent
- Built-in model serialization
- Dataset loading
- Command-line training configuration
- GPU acceleration
- Automatic differentiation
- Extensive runtime error reporting

These are extension opportunities rather than requirements of the core design.

## API summary

### Matrix API

```c
Matrix *create_matrix(int rows, int cols);
void free_matrix(Matrix *m);
double matrix_get(const Matrix *m, int i, int j);
void matrix_set(Matrix *m, int i, int j, double value);
Matrix *copy_matrix(const Matrix *m);
Matrix *add_matrix(const Matrix *m1, const Matrix *m2);
Matrix *multi_matrix(const Matrix *m1, const Matrix *m2);
Matrix *transpose_matrix(const Matrix *m);
Matrix *scale_matrix(const Matrix *m, double scale);
Matrix *subtract_matrix(const Matrix *m1, const Matrix *m2);
Matrix *hadamard_matrix(const Matrix *m1, const Matrix *m2);
Matrix *matrix_apply(const Matrix *m, double (*func)(double));
void matrix_print(const Matrix *m);
void matrix_fill_random(Matrix *m, double min, double max);
```

### Layer API

```c
Layer *create_layer(
    int input_size,
    int output_size,
    double (*activation)(double),
    double (*activation_derivative)(double)
);

void free_layer(Layer *layer);
Matrix *layer_forward(Layer *layer, Matrix *x);
Matrix *layer_backward(
    Layer *layer,
    const Matrix *dL_da,
    double learning_rate
);
```

### Network API

```c
Network *network_create(
    const int *sizes,
    int num_layers,
    double (**activations)(double),
    double (**activation_derivatives)(double)
);

void network_free(Network *network);
Matrix *network_forward(Network *network, Matrix *input);
void network_backward(
    Network *network,
    const Matrix *target,
    double learning_rate
);
```

## License

No license file is currently included. Add an appropriate license before distributing the project or incorporating it into another project.
