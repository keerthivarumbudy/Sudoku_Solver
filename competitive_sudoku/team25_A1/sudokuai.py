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
        N = game_state.board.N
        legal_moves = []
        for i in range(N):
            for j in range(N):
                if game_state.board.get(i, j) == SudokuBoard.empty:
                    for value in range(1, N + 1):
                        if not TabooMove(i, j, value) in game_state.taboo_moves and self.is_legal(game_state, i, j,
                                                                                                  value):
                            legal_moves.append(Move(i, j, value))

        return legal_moves

    def is_legal(self, game_state: GameState, i: int, j: int, value: int):
        N = game_state.board.N
        m = game_state.board.m
        n = game_state.board.n
        for k in range(N):
            if game_state.board.get(k, j) == value:
                return False
            if game_state.board.get(i, k) == value:
                return False

        block_row = i // m
        block_column = j // n
        for k in range(block_row * m, block_row * m + m):
            for o in range(block_column * n, block_column * n + n):
                if game_state.board.get(k, o) == value:
                    return False
        return True

    #   This compute how many points we get by a certain move in a certain board state:
    def compute_move_score(self, board, move):
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
            if board.get(i, move.j) == SudokuBoard.empty and i != move.i:
                break
        else:
            count += 1

        for j in range(0, N):
            if board.get(move.i, j) == SudokuBoard.empty and j != move.j:
                break
        else:
            count += 1

        block_range = [move.i // m, move.j // n]

        for k in range(block_range[0] * m, block_range[0] * m + m):
            counter = 0
            for o in range(block_range[1] * n, block_range[1] * n + n):
                if board.get(k, o) == SudokuBoard.empty and (k, o) != (move.i, move.j):
                    counter = 1
                    break
            if counter == 1:
                break
        else:
            count += 1

        score = score_dict[count]
        return score

    # eval is the difference in points between two players
    def evaluation_function(self, game_state: GameState):
        diff_score = game_state.scores[self.player_number - 1] - game_state.scores[self.opponent_number - 1]
        eval_value = diff_score
        # ----------------------------- maybe try a sophisticated method ------------------------------
        #
        #   ------ the code below just adds another depth, not really sophisticated ------
        #
        # moves = self.find_legal_moves(game_state)
        # a = 1 if self.player_number - len(game_state.moves) % 2 == 1 else -1
        # eval_value = diff_score - a * max([self.compute_move_score(game_state.board, i) for i in moves])
        #
        # -----------------------------------       End       -----------------------------------------
        return eval_value

    # This function check if a board is completely filled
    def is_board_full(self, game_state: GameState):
        N = game_state.board.N
        for i in range(N):
            for j in range(N):
                if game_state.board.get(i, j) == SudokuBoard.empty:
                    return False
        return True

    def minimax_alpha_beta(self, game_state: GameState, max_depth, current_depth, alpha, beta):
        best_move = None
        if current_depth == max_depth or self.is_board_full(game_state):
            return self.evaluation_function(game_state)
        #print("current_depth ", current_depth)
        moves = self.find_legal_moves(game_state)

        if (self.player_number - len(game_state.moves)) % 2 == 1:  # our turn
            if len(moves) == 0:  # if AI_agent cannot find one legal move
                return math.inf  # because there is a taboo in recursion

            current_max = -math.inf   # default current_max = negative inf
            for move in moves:  # find a move maximize the min
                # print("I am thinking the move:"+str(move)+"in depth:"+str(current_depth))
                #   copy and update game_state for next depth of each move:
                new_game_state = copy.deepcopy(game_state)
                new_game_state.board.put(move.i, move.j, move.value)
                new_game_state.scores[self.player_number - 1] += self.compute_move_score(game_state.board, move)
                new_game_state.moves.append(move)
                eval_value = self.minimax_alpha_beta(new_game_state, max_depth, current_depth + 1, alpha, beta)
                # print("eval is:"+str(b))
                #   compare the best current_max and the eval_value of current move:
                #print("current_depth ", current_depth)
                if current_depth == 0 and eval_value > current_max:    # propose a current best move in depth=0
                    best_move = move
                    if max_depth == 1:
                        self.propose_move(move)

                current_max = max(current_max, eval_value)
                if current_max >= beta:
                    return current_max
                if current_max > alpha:
                    alpha = current_max
            if current_depth == 0:
                self.propose_move(best_move)
            return current_max

        else:   # opponent's turn
            if len(moves) == 0:  # if AI_agent cannot find one legal move
                return -math.inf  # because there is a taboo in recursion

            # print("now I think as opponent:")
            current_min = math.inf    # default current_min = positive inf
            for move in moves:  # find a move minimize the max
                # print("if opponent take move:"+str(move)+" in depth:"+str(current_depth))
                #   copy and update game_state for next depth of each move:
                new_game_state = copy.deepcopy(game_state)
                new_game_state.board.put(move.i, move.j, move.value)
                new_game_state.scores[self.opponent_number - 1] += self.compute_move_score(game_state.board, move)
                new_game_state.moves.append(move)
                #   compare the current_min and the eval_value of current move:
                current_min = min(current_min, self.minimax_alpha_beta(new_game_state, max_depth, current_depth + 1,alpha, beta))
                # print("my advantage so far is:"+str(a)+" points")
            # print("I think after opponent moved, my advantage is:"+str(a)+" points")
                if current_min <= alpha:
                    return current_min
                if current_min < beta:
                    beta = current_min
            return current_min

    def compute_best_move(self, game_state: GameState) -> None:

        # -------- First propose a random moves in case we run out of time --------
        #
        # moves = self.find_legal_moves(game_state)
        # self.propose_move(random.choice(moves))
        #
        # ----------------------------    End     ---------------------------------

        # To know the order of our AI_agent player and opponent player:
        [self.player_number, self.opponent_number] = (1, 2) if len(game_state.moves) % 2 == 0 else (2, 1)
        # print("our player_number is:" + str(self.player_number))
        depth = 0   # set the max_depth for minimax()
        while True:
            # run the minimax()
            self.minimax_alpha_beta(game_state, depth + 1, 0, -math.inf, math.inf)
