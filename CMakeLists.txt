cmake_minimum_required(VERSION 3.20)
project(NeuralNetworkInC C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

add_compile_options(-Wall -Wextra -Werror)

# --- Core library: all the actual NN/math code lives here ---
file(GLOB CORE_SOURCES "src/*.c")
add_library(nncore STATIC ${CORE_SOURCES}
        src/activation.c
        include/activation.h
        src/layer.c
        include/layer.h)
target_include_directories(nncore PUBLIC include)
target_link_libraries(nncore PUBLIC m)

# --- Apps: each is its own executable, all linking against nncore ---
add_executable(xor apps/xor_main.c)
target_link_libraries(xor PRIVATE nncore)

# --- Tests ---
enable_testing()
add_executable(test_matrix tests/test_matrix.c)
target_link_libraries(test_matrix PRIVATE nncore)
add_test(NAME test_matrix COMMAND test_matrix)
add_executable(test_activation tests/test_activation.c)
target_link_libraries(test_activation PRIVATE nncore)
add_test(NAME test_activation COMMAND test_activation)
