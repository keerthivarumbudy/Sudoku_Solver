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
    found_taboo = False
    exp_taboo = Move(0, 0, 0)
    num_empty_cells = 0
    def __init__(self):
        super().__init__()

    def find_legal_moves(self, game_state: GameState) -> [Move]:
        N = game_state.board.N
        legal_moves = []
        for i in range(N):
            legal_moves.append([])
            for j in range(N):
                legal_moves.__getitem__(i).append(set())
                if game_state.board.get(i, j) == SudokuBoard.empty:
                    self.num_empty_cells += 1
                    for value in range(1, N + 1):
                        if not TabooMove(i, j, value) in game_state.taboo_moves and self.is_legal(game_state, i, j,
                                                                                                  value):
                            legal_moves.__getitem__(i).__getitem__(j).add(value)

        forced_moves = self.check_forced_move(legal_moves)
        old_forced_moves = []

        counter = 0
        while len(forced_moves) > len(old_forced_moves) and counter < 2:
            for forced_move in forced_moves:
                if forced_move not in old_forced_moves:
                    legal_moves = self.update_lm(game_state, legal_moves, forced_move)
                    legal_moves.__getitem__(forced_move.i).__getitem__(forced_move.j).add(forced_move.value)
                    old_forced_moves.append(forced_move)
            forced_moves = self.check_forced_move(legal_moves)
            counter += 1
        if self.found_taboo:
            legal_moves.__getitem__(self.exp_taboo.i).__getitem__(self.exp_taboo.j).add(self.exp_taboo.value)

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

    def l2a(self, legal_moves):
        """
            Function to change data structure for legal moves
        """

        new_legal = []
        for i in range(len(legal_moves)):
            for j in range(len(legal_moves.__getitem__(i))):
                if len(legal_moves.__getitem__(i).__getitem__(j)) != 0:
                    for value in legal_moves.__getitem__(i).__getitem__(j):
                        new_legal.append(Move(i, j, value))
        return new_legal

    def update_lm(self, game_state, legal_moves, move):
        """
            Function to update legal moves when going into next depth minimax
        """
        N = game_state.board.N
        m = game_state.board.m
        n = game_state.board.n
        moves = legal_moves
        for i in range(N):
            if move.value in moves.__getitem__(i).__getitem__(move.j):
                moves.__getitem__(i).__getitem__(move.j).remove(move.value)
                if self.found_taboo is False and i != move.i:
                    self.exp_taboo = Move(i, move.j, move.value)
                    self.found_taboo = True
            if move.value in moves.__getitem__(move.i).__getitem__(i):
                moves.__getitem__(move.i).__getitem__(i).remove(move.value)
                if self.found_taboo is False and i != move.j:
                    self.exp_taboo = Move(move.i, i, move.value)
                    self.found_taboo = True

        block_row = move.i // m
        block_column = move.j // n
        for k in range(block_row * m, block_row * m + m):
            for o in range(block_column * n, block_column * n + n):
                if move.value in moves.__getitem__(k).__getitem__(o):
                    moves.__getitem__(k).__getitem__(o).remove(move.value)
                    if self.found_taboo is False and k != move.i and o != move.j:
                        self.exp_taboo = Move(k, o, move.value)
                        self.found_taboo = True
        return moves

    def check_forced_move(self, legal_moves):
        forced_moves = []
        for i in range(len(legal_moves)):
            for j in range(len(legal_moves.__getitem__(i))):
                possible = legal_moves.__getitem__(i).__getitem__(j)
                if len(possible) == 1:
                    value = possible.pop()
                    possible.add(value)
                    forced_moves.append(Move(i, j, value))
        return forced_moves

    def compute_move_score(self, board, move):
        """
        Function to compute the points we can get by making a certain move given a certain board state
        """
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




    def evaluation_function(self, game_state: GameState, moves, legal_moves):
        """
        Function to calculate the evaluation value
        -- version 1.0 --
        eval_value = diff_score + point_heuristic
        diff_score is the difference of scores between the two players given a gameState
        heuristic_point is based on the current gameState:
            v1.0: heuristic_point is the max point player can get in next move times a coefficient
                    then times 1 or -1 (if opponent's turn)
        """
        diff_score = game_state.scores[self.player_number - 1] - game_state.scores[self.opponent_number - 1]
        eval_value = diff_score
        moves = moves

        if moves:
            curr_player = 1 if game_state.current_player() == self.player_number else -1
            heuristic_point = max([self.compute_move_score(game_state.board, move) for move in moves])
            eval_value = diff_score + 0.2 * curr_player * heuristic_point

        return eval_value

    def is_board_full(self, game_state: GameState):
        """
        Function to check if a board is completely filled
        """
        N = game_state.board.N
        for i in range(N):
            for j in range(N):
                if game_state.board.get(i, j) == SudokuBoard.empty:
                    return False
        return True

    def minimax_alpha_beta(self, game_state: GameState, max_depth, current_depth, alpha, beta, real_diff_score,
                           legal_moves):
        """
        Function to execute minimax algorithm with alpha-beta pruning
        """
        best_move = None
        moves = self.l2a(legal_moves)
        if current_depth == max_depth or self.is_board_full(game_state):
            return self.evaluation_function(game_state, moves, legal_moves)

        if (self.player_number - len(game_state.moves)) % 2 == 1:  # our turn
            if len(moves) == 0:  # if AI_agent cannot find one legal move
                return real_diff_score  # because there is a taboo in recursion

            current_max = -math.inf  # default current_max = negative inf
            for move in moves:  # find a move to maximize the min
                #   copy and update game_state for next depth of each move:
                new_game_state = copy.deepcopy(game_state)
                new_legal_moves = copy.deepcopy(legal_moves)
                new_game_state.board.put(move.i, move.j, move.value)
                new_legal_moves = self.update_lm(game_state, new_legal_moves, move)
                new_game_state.scores[self.player_number - 1] += self.compute_move_score(game_state.board, move)
                new_game_state.moves.append(move)
                eval_value = self.minimax_alpha_beta(new_game_state, max_depth, current_depth + 1,
                                                     alpha, beta, real_diff_score, new_legal_moves)
                #   compare the best current_max and the eval_value of current move
                if current_depth == 0 and eval_value > current_max:  # propose a current best move in depth=0
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

        else:  # opponent's turn
            if len(moves) == 0:  # if AI_agent cannot find one legal move
                return real_diff_score  # because there is a taboo in recursion

            current_min = math.inf  # default current_min = positive inf
            for move in moves:  # find a move minimize the max
                #   copy and update game_state for next depth of each move:
                new_game_state = copy.deepcopy(game_state)
                new_legal_moves = copy.deepcopy(legal_moves)
                new_game_state.board.put(move.i, move.j, move.value)
                new_legal_moves = self.update_lm(game_state, new_legal_moves, move)
                new_game_state.scores[self.opponent_number - 1] += self.compute_move_score(game_state.board, move)
                new_game_state.moves.append(move)
                #   compare the current_min and the eval_value of current move:
                current_min = min(current_min, self.minimax_alpha_beta(new_game_state, max_depth, current_depth + 1,
                                                                       alpha, beta, real_diff_score, new_legal_moves))
                if current_min <= alpha:
                    return current_min
                if current_min < beta:
                    beta = current_min
            return current_min

    def compute_best_move(self, game_state: GameState) -> None:

        # To know the order of our AI_agent player and opponent player:
        [self.player_number, self.opponent_number] = (1, 2) if len(game_state.moves) % 2 == 0 else (2, 1)
        depth = 1  # set the max_depth for minimax()
        initial_legal_moves = self.find_legal_moves(game_state)
        real_diff_score = self.evaluation_function(game_state, self.l2a(initial_legal_moves), initial_legal_moves)

        while True:
            score = self.minimax_alpha_beta(game_state, depth, 0, -math.inf, math.inf, real_diff_score, initial_legal_moves)
            # During early game, check whether we get the last move or not
            if self.num_empty_cells > game_state.board.m + game_state.board.n:
                if self.num_empty_cells % 2 == 0:
                    # we do not get last move
                    if score <= 0 and self.found_taboo:
                        self.propose_move(self.exp_taboo)
                        return
            depth += 1
