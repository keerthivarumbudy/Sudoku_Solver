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

    def __init__(self):
        super().__init__()

    def find_legal_moves(self, game_state: GameState) -> [Move]:
        """
            Function to construct or load and update the data structure for legal moves
        """
        N = game_state.board.N
        m = game_state.board.m
        n = game_state.board.n
        # check if there already exists a matrix of suggested legal moves
        legal_moves = self.load()
        if legal_moves is None:
            # if there is no matrix saved, construct a new one using the gamestate
            legal_moves = self.construct_legal(game_state)
        else:
            # if a matrix exists it needs to be updated using the previously played moves
            history = 1
            if len(game_state.moves) >= 2:
                # make sure more than 1 moves have been played
                history = 2
            for k in range(history):
                # for the past 1 or 2 moves update the matrix
                move = game_state.moves[-(k+1)]
                if move not in game_state.taboo_moves:
                    # if the move is not a taboo move a normal update can be used
                    legal_moves = self.update_lm(game_state, legal_moves, move)
                elif move.value in legal_moves.__getitem__(move.i).__getitem__(move.j):
                    # else if it is a taboo move and the value is still in the matrix remove it
                    legal_moves.__getitem__(move.i).__getitem__(move.j).remove(move.value)
            self.found_taboo = False
            self.exp_taboo = Move(0, 0, 0)

        # Use the heuristic for forced moves to reduce the amount of legal moves
        legal_moves = self.find_forced_move(game_state, legal_moves)

        # Use the heuristic for outlier moves to reduce the amount of legal moves
        legal_moves = self.find_outlier_move(game_state, legal_moves)

        if self.found_taboo:
            # If a taboo move has been found make sure it is added to the legal moves
            legal_moves.__getitem__(self.exp_taboo.i).__getitem__(self.exp_taboo.j).add(self.exp_taboo.value)
        # Save the matrix of legal moves for the next turn
        self.save(legal_moves)
        return legal_moves

    def construct_legal(self, game_state):
        """
            Function to construct the data structure for legal moves
        """
        N = game_state.board.N
        # legal_moves is a matrix with the same size as the board
        legal_moves = []
        # For each cell on the sudoku board create a set in the matrix
        for i in range(N):
            legal_moves.append([])
            for j in range(N):
                legal_moves.__getitem__(i).append(set())
                if game_state.board.get(i, j) == SudokuBoard.empty:
                    # If a cell on the board is empty fill the set with all possible legal values for this place
                    for value in range(1, N + 1):
                        if not TabooMove(i, j, value) in game_state.taboo_moves and self.is_legal(game_state, i, j,
                                                                                                  value):
                            legal_moves.__getitem__(i).__getitem__(j).add(value)
        return legal_moves

    def is_legal(self, game_state: GameState, i: int, j: int, value: int):
        """
            Function to check if a move is legal
        """
        N = game_state.board.N
        m = game_state.board.m
        n = game_state.board.n
        # Check if the value already exists in the same row and/or column
        for k in range(N):
            if game_state.board.get(k, j) == value:
                return False
            if game_state.board.get(i, k) == value:
                return False

        block_row = i // m
        block_column = j // n
        # Check if the value already exists in the same block
        for k in range(block_row * m, block_row * m + m):
            for o in range(block_column * n, block_column * n + n):
                if game_state.board.get(k, o) == value:
                    return False
        return True

    def l2a(self, legal_moves):
        """
            Function to change data structure for legal moves into an array
        """

        new_legal = []
        # Create an array where each value is a move constructed using the indices and values of the sets
        for i in range(len(legal_moves)):
            for j in range(len(legal_moves.__getitem__(i))):
                if len(legal_moves.__getitem__(i).__getitem__(j)) != 0:
                    for value in legal_moves.__getitem__(i).__getitem__(j):
                        new_legal.append(Move(i, j, value))
        return new_legal

    def update_lm(self, game_state, legal_moves, move):
        """
            Function to update legal moves
        """
        N = game_state.board.N
        m = game_state.board.m
        n = game_state.board.n
        moves = legal_moves
        # Make sure the set for the proposed move is cleared
        moves.__getitem__(move.i).__getitem__(move.j).clear()
        moves.__getitem__(move.i).__getitem__(move.j).add(move.value)
        # Remove the move.value from any set in the same row or column
        for i in range(N):
            if move.value in moves.__getitem__(i).__getitem__(move.j):
                moves.__getitem__(i).__getitem__(move.j).remove(move.value)
                if self.found_taboo is False and i != move.i:
                    # If no taboo move has been found this is stored as a taboo move
                    self.exp_taboo = Move(i, move.j, move.value)
                    self.found_taboo = True
            if move.value in moves.__getitem__(move.i).__getitem__(i):
                moves.__getitem__(move.i).__getitem__(i).remove(move.value)
                if self.found_taboo is False and i != move.j:
                    # If no taboo move has been found this is stored as a taboo move
                    self.exp_taboo = Move(move.i, i, move.value)
                    self.found_taboo = True

        block_row = move.i // m
        block_column = move.j // n
        # Remove the move.value from any set in the same block
        for k in range(block_row * m, block_row * m + m):
            for o in range(block_column * n, block_column * n + n):
                if move.value in moves.__getitem__(k).__getitem__(o):
                    moves.__getitem__(k).__getitem__(o).remove(move.value)
                    if self.found_taboo is False and k != move.i and o != move.j:
                        # If no taboo move has been found this is stored as a taboo move
                        self.exp_taboo = Move(k, o, move.value)
                        self.found_taboo = True
        return moves

    def find_forced_move(self, game_state, legal_moves):
        """
            Function to find and use forced moves
        """
        # Find all moves that exist as an only value in a set
        forced_moves = self.check_forced_move(legal_moves)
        old_forced_moves = []

        counter = 0
        # While new forced moves have still been found and the counter is not 3
        while len(forced_moves) > len(old_forced_moves) and counter < 3:
            # Update the matrix assuming each forced move will eventually be played
            for forced_move in forced_moves:
                if forced_move not in old_forced_moves:
                    legal_moves = self.update_lm(game_state, legal_moves, forced_move)
                    # Make sure the proposed move is still in the matrix and update old_forced_moves
                    legal_moves.__getitem__(forced_move.i).__getitem__(forced_move.j).add(forced_move.value)
                    old_forced_moves.append(forced_move)
            forced_moves = self.check_forced_move(legal_moves)
            counter += 1
        return legal_moves

    def check_forced_move(self, legal_moves):
        """
            Function to find forced moves
        """
        forced_moves = []
        # Check each set and add all values that exist in a set alone
        for i in range(len(legal_moves)):
            for j in range(len(legal_moves.__getitem__(i))):
                possible = legal_moves.__getitem__(i).__getitem__(j)
                if len(possible) == 1:
                    value = possible.pop()
                    possible.add(value)
                    forced_moves.append(Move(i, j, value))
        return forced_moves

    def find_outlier_move(self, game_state, legal_moves):
        """
            Function to find and use outlier moves
        """
        N = game_state.board.N
        m = game_state.board.m
        n = game_state.board.n

        counter = 0
        num_legal_moves = len(self.l2a(legal_moves))
        while counter < 2 and num_legal_moves > (N*N*N)/10:
            for i in range(N):
                row = [-1] * N
                column = [-1] * N
                for j in range(N):
                    for value in legal_moves.__getitem__(i).__getitem__(j):
                        if row[value - 1] == -1:
                            row[value - 1] = j
                        else:
                            row[value - 1] = -2
                    for value in legal_moves.__getitem__(j).__getitem__(i):
                        if column[value - 1] == -1:
                            column[value - 1] = j
                        else:
                            column[value - 1] = -2
                for j in range(N):
                    if row[j] >= 0:
                        self.update_lm(game_state,legal_moves,Move(i, row[j], j+1))
                        legal_moves.__getitem__(i).__getitem__(row[j]).add(j+1)
                    if column[j] >= 0:
                        self.update_lm(game_state,legal_moves,Move(column[j], i, j+1))
                        legal_moves.__getitem__(column[j]).__getitem__(i).add(j+1)
            for block_row in range(m):
                for block_column in range(n):
                    block = [-1] * N
                    for k in range(block_row * m, block_row * m + m):
                        for o in range(block_column * n, block_column * n + n):
                            for value in legal_moves.__getitem__(k).__getitem__(o):
                                if block[value - 1] == -1:
                                    block[value - 1] = (k, o)
                                else:
                                    block[value - 1] = -2
                    for j in range(N):
                        if type(block[j]) is tuple:
                            self.update_lm(game_state,legal_moves,Move(block[j][0], block[j][1], j + 1))
                            legal_moves.__getitem__(block[j][0]).__getitem__(block[j][1]).add(j + 1)

            counter += 1
        return legal_moves

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
            self.minimax_alpha_beta(game_state, depth, 0, -math.inf, math.inf, real_diff_score, initial_legal_moves)
            depth += 1
