#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

import copy
import math
import random

import competitive_sudoku.sudokuai
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove


class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    Sudoku AI that computes a move for a given sudoku configuration.
    """
    player_number = 1
    opponent_number = 2

    def __init__(self):
        super().__init__()

    def find_legal_moves(self, game_state: GameState) -> [Move]:
        # function to find all legal moves for a certain game state
        N = game_state.board.N
        legal_moves = []
        for i in range(N):
            for j in range(N):
                # for each position on the board check if it is an empty position
                if game_state.board.get(i, j) == SudokuBoard.empty:
                    # if it is empty iterate ove each possible value
                    for value in range(1, N + 1):
                        # only if this tuple is legal and not taboo add it to the legal moves
                        if not TabooMove(i, j, value) in game_state.taboo_moves and self.is_legal(game_state, i, j,
                                                                                                  value):
                            legal_moves.append(Move(i, j, value))

        return legal_moves

    def is_legal(self, game_state: GameState, i: int, j: int, value: int):
        # function to determine if a position and value tuple is legal
        N = game_state.board.N
        m = game_state.board.m
        n = game_state.board.n
        for k in range(N):
            # for each value in the same row or column as the tuple check if new value is present
            if game_state.board.get(k, j) == value:
                return False
            if game_state.board.get(i, k) == value:
                return False

        block_row = i // m      # Determine in which block the tuple belongs
        block_column = j // n
        for k in range(block_row * m, block_row * m + m):
            for o in range(block_column * n, block_column * n + n):
                # # for each value in the same block as the tuple check if new value is present
                if game_state.board.get(k, o) == value:
                    return False
        return True

    def compute_move_score(self, board, move):
        # Function to compute the points we can get by making a certain move given a certain board state
        score_dict = {
            0: 0,
            1: 1,
            2: 3,
            3: 7
        }

        m = board.m
        n = board.n
        N = board.N

        count = 0
        for i in range(0, N):
            # check if current row is full
            if board.get(i, move.j) == SudokuBoard.empty and i != move.i:
                break
        else:
            count += 1

        for j in range(0, N):
            # check if current column is full
            if board.get(move.i, j) == SudokuBoard.empty and j != move.j:
                break
        else:
            count += 1

        block_range = [move.i // m, move.j // n]

        for k in range(block_range[0] * m, block_range[0] * m + m):
            counter = 0
            for o in range(block_range[1] * n, block_range[1] * n + n):
                # check if current block is full
                if board.get(k, o) == SudokuBoard.empty and (k, o) != (move.i, move.j):
                    counter = 1
                    break
            if counter == 1:
                break
        else:
            count += 1

        score = score_dict[count]
        return score


    def evaluation_function(self, game_state: GameState):
        # Function to calculate the difference of scores between the two players given a gamestate
        diff_score = game_state.scores[self.player_number - 1] - game_state.scores[self.opponent_number - 1]
        eval_value = diff_score
        return eval_value

    def is_board_full(self, game_state: GameState):
        # Function to check if a board is completely filled
        N = game_state.board.N
        for i in range(N):
            for j in range(N):
                if game_state.board.get(i, j) == SudokuBoard.empty:
                    return False
        return True

    def minimax_alpha_beta(self, game_state: GameState, max_depth, current_depth, alpha, beta, real_diff_score):
        # Function to execute minimax algorithm with alpha-beta pruning
        best_move = None
        if current_depth == max_depth or self.is_board_full(game_state):
            return self.evaluation_function(game_state)
        moves = self.find_legal_moves(game_state)

        if (self.player_number - len(game_state.moves)) % 2 == 1:  # our turn
            if len(moves) == 0:  # if AI_agent cannot find one legal move
                return real_diff_score  # because there is a taboo in recursion

            current_max = -math.inf  # default current_max = negative inf
            for move in moves:  # find a move to maximize the min
                #   copy and update game_state for next depth of each move:
                new_game_state = copy.deepcopy(game_state)
                new_game_state.board.put(move.i, move.j, move.value)
                new_game_state.scores[self.player_number - 1] += self.compute_move_score(game_state.board, move)
                new_game_state.moves.append(move)
                eval_value = self.minimax_alpha_beta(new_game_state, max_depth, current_depth + 1,
                                                     alpha, beta, real_diff_score)
                #   compare the best current_max and the eval_value of current move
                if current_depth == 0 and eval_value > current_max:  # propose a current best move in depth=0
                    best_move = move
                    if max_depth == 1:
                        self.propose_move(move)

                current_max = max(current_max, eval_value)
                # beta pruning
                if current_max >= beta:
                    return current_max
                if current_max > alpha:
                    alpha = current_max
            if current_depth == 0:
                self.propose_move(best_move)
            return current_max

        else:  # opponent's turn
            if len(moves) == 0:  # if AI_agent cannot find one legal move
                return real_diff_score  # because there is a taboo in recursion

            current_min = math.inf  # default current_min = positive inf
            for move in moves:  # find a move minimize the max
                #   copy and update game_state for next depth of each move:
                new_game_state = copy.deepcopy(game_state)
                new_game_state.board.put(move.i, move.j, move.value)
                new_game_state.scores[self.opponent_number - 1] += self.compute_move_score(game_state.board, move)
                new_game_state.moves.append(move)
                #   compare the current_min and the eval_value of current move:
                current_min = min(current_min, self.minimax_alpha_beta(new_game_state, max_depth, current_depth + 1,
                                                                       alpha, beta, real_diff_score))
                # alpha pruning
                if current_min <= alpha:
                    return current_min
                if current_min < beta:
                    beta = current_min
            return current_min

    def compute_best_move(self, game_state: GameState) -> None:

        # To know the order of our AI_agent player and opponent player:
        [self.player_number, self.opponent_number] = (1, 2) if len(game_state.moves) % 2 == 0 else (2, 1)
        depth = 1  # set the max_depth for minimax()
        real_diff_score = self.evaluation_function(game_state)
        while True:
            # run the minimax() with iterative deepening
            self.minimax_alpha_beta(game_state, depth, 0, -math.inf, math.inf, real_diff_score)
            depth += 1
