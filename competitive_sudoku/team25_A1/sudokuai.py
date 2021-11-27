#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

import random
import time
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
import competitive_sudoku.sudokuai
import copy


class Node:
    def __init__(self, move, score):
        self.children = None
        self.move = move
        self.score = score

    def set_children(self, children):
        # not sure if this is needed, but can be used to trace it back I guess
        self.children = children


class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    Sudoku AI that computes a move for a given sudoku configuration.
    """

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

    def evaluate_move(self, board, move):
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

    def get_min_max(self, depth, children):
        # depending on which player's turn it is, selects and returns min or max score of the set of moves
        if depth % 2 != 0:
            maximum = Node(None, -1)
            for i in children:
                if i.score > maximum.score:
                    maximum = i
            print("max=", maximum.score, " in depth=", depth)
            return maximum
        else:
            minimum = Node(None, 8)
            for i in children:
                if i.score < minimum.score:
                    minimum = i
            print("min=", minimum.score, " in depth=", depth)
            return minimum

    def minimax(self, parent, game_state, max_depth, current_depth):
        legal_moves = self.find_legal_moves(game_state)
        final_branch = None
        maximum = -1000
        minimum = 1000
        def possible(i, j, value):
            return game_state.board.get(i, j) == SudokuBoard.empty and not TabooMove(i, j,
                                                                                     value) in game_state.taboo_moves

        all_moves = [move for move in legal_moves if possible(move.i, move.j, move.value)]
        # print("current depth = ", current_depth, "max_depth = ", max_depth)
        # for move in all_moves:
        #     print(move.i, move.j, move.value)
        if current_depth == max_depth:
            # ends the recursion when max-depth is reached, picks the min or max to return based on turn
            return parent

        if len(all_moves) == 0:
            # ends recursion when last node is reached
            return parent

        for move in all_moves:
            # assigns score to each mode, changes copy of the board based on the move, runs recursion to create and
            # traverse the tree
            score = self.evaluate_move(game_state.board, move)
            game_state_copy = copy.deepcopy(game_state)
            game_state_copy.board.put(move.i,move.j,move.value)
            if current_depth % 2 == 0:
                node = Node(move, parent.score + score)
                branch = self.minimax(node, game_state_copy, max_depth, current_depth + 1)
                if branch.score > maximum:
                    maximum = branch.score
                    final_branch = branch
            else:
                node = Node(move, parent.score - score)
                branch = self.minimax(node, game_state_copy, max_depth, current_depth + 1)
                if branch.score < minimum:
                    minimum = branch.score
                    final_branch = branch
        return final_branch

    def compute_best_move(self, game_state: GameState) -> None:
        depth = 1
        move = self.minimax(Node(None, 0), game_state, depth, 0).move
        self.propose_move(move)

        while True:
            time.sleep(0.1)
            depth += 1
            move = self.minimax(Node(None, 0), game_state, depth, 0).move
            self.propose_move(move)
