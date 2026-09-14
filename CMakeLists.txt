cmake_minimum_required(VERSION 3.20)
project(NeuralNetworkInC C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

add_compile_options(-Wall -Wextra -Werror)

# --- Core library: all the actual NN/math code lives here ---
file(GLOB CORE_SOURCES "src/*.c")
add_library(nncore STATIC ${CORE_SOURCES}
        src/matrix.c
        include/matrix.h
        src/activation.c
        include/activation.h
        src/layer.c
        include/layer.h
        src/network.c
        include/network.h
        src/loss.c
        include/loss.h
        src/tensor.c
        include/tensor.h
        src/ops.c
        include/ops.h
        src/optim.c
        include/optim.h

)
target_include_directories(nncore PUBLIC include)
target_link_libraries(nncore PUBLIC m)

# --- Apps: each is its own executable, all linking against nncore ---
add_executable(xor apps/xor_main.c)
target_link_libraries(xor PRIVATE nncore)
add_executable(rps apps/rps_main.c)
target_link_libraries(rps PRIVATE nncore)

# --- Tests ---
enable_testing()
add_executable(test_matrix tests/test_matrix.c)
target_link_libraries(test_matrix PRIVATE nncore)
add_test(NAME test_matrix COMMAND test_matrix)
add_executable(test_activation tests/test_activation.c)
target_link_libraries(test_activation PRIVATE nncore)
add_test(NAME test_activation COMMAND test_activation)
add_executable(test_layer tests/test_layer.c)
target_link_libraries(test_layer PRIVATE nncore)
add_test(NAME test_layer COMMAND test_layer)
add_executable(test_network tests/test_network.c)
target_link_libraries(test_network PRIVATE nncore)
add_test(NAME test_network COMMAND test_network)
add_executable(test_loss tests/test_loss.c)
target_link_libraries(test_loss PRIVATE nncore)
add_test(NAME test_loss COMMAND test_loss)
add_executable(test_layer_backward tests/test_layer_backward.c)
target_link_libraries(test_layer_backward PRIVATE nncore)
add_test(NAME test_layer_backward COMMAND test_layer_backward)
add_executable(test_tensor tests/test_tensor.c)
target_link_libraries(test_tensor PRIVATE nncore)
add_test(NAME test_tensor COMMAND test_tensor)