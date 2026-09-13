#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>

#include "activation.h"
#include "loss.h"
#include "matrix.h"
#include "network.h"

static const char *move_name(const int move) {
    static const char *names[] = {"rock", "paper", "scissors"};
    return names[move];
}

static int read_move(void) {
    char line[32];

    if (!fgets(line, sizeof(line), stdin)) {
        return -1;
    }

    switch (tolower((unsigned char)line[0])) {
        case 'r':
            return 0;
        case 'p':
            return 1;
        case 's':
            return 2;
        case 'q':
            return 3;
        default:
            return -1;
    }
}

static int best_move(const Matrix *output) {
    int move = 0;

    for (int i = 1; i < output->rows; ++i) {
        if (matrix_get(output, i, 0) > matrix_get(output, move, 0)) {
            move = i;
        }
    }
    return move;
}

int main(void) {
    /*
     * The training examples teach the network the winning response:
     * rock -> paper, paper -> scissors, and scissors -> rock.
     */
    const double input_values[3][3] = {
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0}
    };
    const double target_values[3][3] = {
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0},
        {1.0, 0.0, 0.0}
    };

    Matrix *inputs[3];
    Matrix *targets[3];
    for (int i = 0; i < 3; ++i) {
        inputs[i] = create_matrix(3, 1);
        targets[i] = create_matrix(3, 1);
        for (int j = 0; j < 3; ++j) {
            matrix_set(inputs[i], j, 0, input_values[i][j]);
            matrix_set(targets[i], j, 0, target_values[i][j]);
        }
    }

    srand(42);
    const int sizes[] = {3, 6, 3};
    double (*activations[])(double) = {sigmoid, sigmoid};
    double (*activation_derivatives[])(double) = {
        sigmoid_derivative,
        sigmoid_derivative
    };
    Network *network = network_create(
        sizes,
        2,
        activations,
        activation_derivatives
    );

    const int epochs = 20000;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (int i = 0; i < 3; ++i) {
            const double learning_rate = 0.7;
            network_forward(network, inputs[i]);
            network_backward(network, targets[i], learning_rate);
        }
    }

    printf("Rock Paper Scissors\n");
    printf("Enter r for rock, p for paper, s for scissors, or q to quit.\n\n");

    int player_score = 0;
    int computer_score = 0;
    while (1) {
        printf("Your move: ");
        const int player_move = read_move();
        if (player_move == 3) {
            break;
        }
        if (player_move < 0) {
            printf("Please enter r, p, s, or q.\n");
            continue;
        }

        Matrix *output = network_forward(network, inputs[player_move]);
        const int computer_move = best_move(output);

        printf("You played %s; the network played %s.\n",
               move_name(player_move), move_name(computer_move));

        if (player_move == computer_move) {
            printf("Draw!\n");
        } else if ((computer_move - player_move + 3) % 3 == 1) {
            ++computer_score;
            printf("The network wins!\n");
        } else {
            ++player_score;
            printf("You win!\n");
        }
        printf("Score: you %d, network %d\n\n", player_score, computer_score);
    }

    for (int i = 0; i < 3; ++i) {
        free_matrix(inputs[i]);
        free_matrix(targets[i]);
    }
    network_free(network);
    printf("Thanks for playing!\n");
    return 0;
}
