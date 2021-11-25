#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

import random
import time
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
import competitive_sudoku.sudokuai


class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    Sudoku AI that computes a move for a given sudoku configuration.
    """

    def __init__(self):
        super().__init__()

    def evaluation_function(self, game_state, move):
        score_dict = {
            0 : 0,
            1 : 1,
            2 : 3,
            3 : 7
        }
        m = game_state.board.m
        n = game_state.board.n
        N = game_state.board.N
        count = 0
        for i in range(0, N):
           if game_state.board.get(i,move.j)== SudokuBoard.empty:
                break
        else:
            count += 1

        for j in range(0, N):
           if game_state.board.get(move.i,j)== SudokuBoard.empty:
                break
        else:
            count += 1

        block_range = [i // m, j// n ]

        for k in range(block_range[0]*m, block_range[0]*m + m):
            for o in range(block_range[1]*n, block_range[1]*n + n):
                if game_state.board.get(k, o) == SudokuBoard.empty:
                   break
        else:
            count += 1


        score = score_dict[count]
        return score

    # N.B. This is a very naive implementation.
    def compute_best_move(self, game_state: GameState) -> None:
        N = game_state.board.N

        def possible(i, j, value):
            return game_state.board.get(i, j) == SudokuBoard.empty and not TabooMove(i, j, value) in game_state.taboo_moves

        all_moves = [Move(i, j, value) for i in range(N) for j in range(N) for value in range(1, N+1) if possible(i, j, value)]
        move = random.choice(all_moves)
        self.propose_move(move)
        print(self.evaluation_function(game_state, move))
        while True:
            time.sleep(0.2)
            self.propose_move(random.choice(all_moves))



