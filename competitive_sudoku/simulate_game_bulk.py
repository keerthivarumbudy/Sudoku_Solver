#!/usr/bin/env python3

#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)
"""Usage: 
    python  simulate_game_bulk.py --first=team19_A1 --second=greedy_player --board=boards/hard-3x3.txt  --iter=20 --workers=12 --time=.5
    python  simulate_game_bulk.py --second=team19_A1 --first=greedy_player --board=boards/hard-3x3.txt  --iter=20 --workers=12 --time=.5

    python  simulate_game_bulk.py --first=team19_A1 --second=greedy_player --board=boards\empty-3x3.txt --time=.5 --iter=30 --workers=12
    python  simulate_game_bulk.py --first=greedy_player --second=team19_A1  --board=boards\empty-3x3.txt --time=.5 --iter=30 --workers=12

    python simulate_game_bulk.py --first=team19_A1 --second=greedy_player --board=boards/empty-4x4.txt --iter=50 --workers=12 --time=.5 >> bulk_empty4x4


     python3  simulate_game_bulk.py --first=team25_A1 --second=greedy_player --board="boards/random-3x3.txt"  --iter=5 --workers=6 --time=5
     python3  simulate_game_bulk.py --first=greedy_player --second=team25_A1 --board="boards/empty-3x3.txt"  --iter=5 --workers=6 --time=10

    """

import argparse
import csv
import importlib
import multiprocessing
from os import sep
import logging
import platform
import re
import time
import concurrent.futures as cf
from pathlib import Path
from typing import Counter
from competitive_sudoku.execute import solve_sudoku
from competitive_sudoku.sudoku import GameState, SudokuBoard, Move, TabooMove, load_sudoku_from_text
from competitive_sudoku.sudokuai import SudokuAI

# Prevent unwanted logging messages from AI module
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.WARNING) 


def check_oracle(solve_sudoku_path: str) -> None:
    board_text = '''2 2
       1   2   3   4
       3   4   .   2
       2   1   .   3
       .   .   .   1
    '''
    output = solve_sudoku(solve_sudoku_path, board_text)
    result = 'has a solution' in output
    if result:
        print('The sudoku_solve program works.')
    else:
        print('The sudoku_solve program gives unexpected results.')
        print(output)


def simulate_game(initial_board: SudokuBoard, player1: SudokuAI, player2: SudokuAI, solve_sudoku_path: str, calculation_time: float = 0.5, match_number=None) -> None:
    """
    Simulates a game between two instances of SudokuAI, starting in initial_board. The first move is played by player1.
    @param initial_board: The initial position of the game.
    @param player1: The AI of the first player.
    @param player2: The AI of the second player.
    @param solve_sudoku_path: The location of the oracle executable.
    @param calculation_time: The amount of time in seconds for computing the best move.
    """
    if match_number is not None:
        print("Started match", match_number)
    import copy
    N = initial_board.N
    

    game_state = GameState(initial_board, copy.deepcopy(initial_board), [], [], [0, 0])
    move_number = 0
    number_of_moves = initial_board.squares.count(SudokuBoard.empty)
    # print('Initial state')
    # print(game_state)

    with multiprocessing.Manager() as manager:
        # use a lock to protect assignments to best_move
        lock = multiprocessing.Lock()
        player1.lock = lock
        player2.lock = lock

        # use shared variables to store the best move
        player1.best_move = manager.list([0, 0, 0])
        player2.best_move = manager.list([0, 0, 0])

        while move_number < number_of_moves:
            player, player_number = (player1, 1) if len(game_state.moves) % 2 == 0 else (player2, 2)
            # print(f'-----------------------------\nCalculate a move for player {player_number}')
            player.best_move[0] = 0
            player.best_move[1] = 0
            player.best_move[2] = 0
            try:
                process = multiprocessing.Process(target=player.compute_best_move, args=(game_state,))
                process.start()
                time.sleep(calculation_time)
                lock.acquire()
                process.terminate()
                lock.release()
            except Exception as err:
                print('Error: an exception occurred.\n', err)
            i, j, value = player.best_move
            best_move = Move(i, j, value)
            # print(f'Best move: {best_move}')
            player_score = 0
            if best_move != Move(0, 0, 0):
                if TabooMove(i, j, value) in game_state.taboo_moves:
                    # print(f'Error: {best_move} is a taboo move. Player {2-player_number} wins the game.')
                    return f"Player {player_number} made TABOO move"
                board_text = str(game_state.board)
                options = f'--move "{game_state.board.rc2f(i, j)} {value}"'
                output = solve_sudoku(solve_sudoku_path, board_text, options)
                if 'Invalid move' in output:
                    # print(f'Error: {best_move} is not a valid move. Player {3-player_number} wins the game.')
                    return f"Player {player_number} made INVALID move"
                if 'Illegal move' in output:
                    # print(f'Error: {best_move} is not a legal move. Player {3-player_number} wins the game.')
                    return f"Player {player_number} made ILLEGAL move"
                if 'has no solution' in output:
                    # print(f'The sudoku has no solution after the move {best_move}.')
                    player_score = 0
                    game_state.moves.append(TabooMove(i, j, value))
                    game_state.taboo_moves.append(TabooMove(i, j, value))
                if 'The score is' in output:
                    match = re.search(r'The score is ([-\d]+)', output)
                    if match:
                        player_score = int(match.group(1))
                        game_state.board.put(i, j, value)
                        game_state.moves.append(best_move)
                        move_number = move_number + 1
                    else:
                        raise RuntimeError(f'Unexpected output of sudoku solver: "{output}".')
            else:
                # print(f'No move was supplied. Player {3-player_number} wins the game.')
                return f"Player {player_number} was too slow"
            game_state.scores[player_number-1] = game_state.scores[player_number-1] + player_score
            # print(f'Reward: {player_score}')
            # print(game_state)
        # print('Final score: %d - %d' % (game_state.scores[0], game_state.scores[1]))
        return game_state.scores


def main():
    solve_sudoku_path = 'bin\\solve_sudoku.exe' if platform.system() == 'Windows' else 'bin/solve_sudoku'

    cmdline_parser = argparse.ArgumentParser(description='Script for simulating a competitive sudoku game.')
    cmdline_parser.add_argument('--first', help="the module name of the first player's SudokuAI class (default: random_player)", default='random_player')
    cmdline_parser.add_argument('--second', help="the module name of the second player's SudokuAI class (default: random_player)", default='random_player')
    cmdline_parser.add_argument('--time', help="the time (in seconds) for computing a move (default: 0.5)", type=float, default=0.5)
    cmdline_parser.add_argument('--check', help="check if the solve_sudoku program works", action='store_true')
    cmdline_parser.add_argument('--board', metavar='FILE', type=str, help='a text file containing the start position')
    cmdline_parser.add_argument('--iter', type=int, default=1, help="number of iterations to execute")
    cmdline_parser.add_argument('--workers', type=int, default=1, help="number of workers used for concurrent bulk solving")
    args = cmdline_parser.parse_args()

    if args.check:
        check_oracle(solve_sudoku_path)
        return

    board_text = '''2 2
       1   2   .   4
       .   4   .   2
       2   1   .   3
       .   .   .   1
    '''
    if args.board:
        board_text = Path(args.board).read_text()
    board = load_sudoku_from_text(board_text)

    module1 = importlib.import_module(args.first + '.sudokuai')
    module2 = importlib.import_module(args.second + '.sudokuai')
    player1 = [module1.SudokuAI() for _ in range(args.iter)]
    player2 = [module2.SudokuAI() for _ in range(args.iter)]
    if args.first in ('random_player', 'greedy_player'):
        for i in range(args.iter):
            player1[i].solve_sudoku_path = solve_sudoku_path
    if args.second in ('random_player', 'greedy_player'):
        for i in range(args.iter):
            player2[i].solve_sudoku_path = solve_sudoku_path
    
    won = 0
    lost = 0
    draw = 0
    mistakes = Counter()
    none = 0

    with cf.ThreadPoolExecutor(args.workers) as executor:
        results = [executor.submit(simulate_game, board, player1[i], player2[i], solve_sudoku_path, args.time, i) for i in range(args.iter)]
        for f in cf.as_completed(results):
            scores = f.result()
            print("A match finished with outcome: ", end='')
            if scores is None:
                none += 1
                print("Scores was None")
                continue
            elif isinstance(scores, str):
                mistakes[scores] += 1
                print(scores)
            else:
                s1 = scores[0]
                s2 = scores[1]
                if s1 > s2:
                    print("Player 1 (%s) won! Score %d - %d" % (args.first, s1, s2))
                    won += 1
                elif s1 < s2:
                    print("Player 2 (%s) won! Score %d - %d" % (args.second, s1, s2))
                    lost += 1
                else: 
                    print("Draw! Score %d - %d" % (s1, s2))
                    draw += 1

    column_width = 27
    spacer = " "*column_width
    print()
    print('━'*(column_width+7))
    print("Scoreboard:")
    print(f"{args.iter} matches")
    print('═'*(column_width+7))
    print(spacer, f"{won/args.iter:>6.1%} \rP1 [ {args.first:^14}] wins:")
    print(spacer, f"{draw/args.iter:>6.1%}\r{'draw: ':>{column_width}}")
    print(spacer, f"{lost/args.iter:>6.1%}\rP2 [ {args.second:^14}] wins:")
    print('╌'*(column_width+7))
    for key, value in mistakes.items():
        print(spacer, f"{value/args.iter:>6.1%}\r{key}:")
    print(spacer, f"{none/args.iter:>6.1%}\rAll else:")
    print('━'*(column_width+7))
    print(
    f"P1:         {args.first}",
    f"P2:         {args.second}",
    f"Board:      {args.board}",
    f"Time:       {args.time}s",
    f"Workers:    {args.workers}",
    f"Iterations: {args.iter}",
    sep='\n'
    )
    push_to_csv(args.board, args.first, args.second, won*100/args.iter, lost*100/args.iter, draw*100/args.iter, args.time)

def push_to_csv(board,player1, player2, p1_wins, p2_wins, draws,time):
    with open('../results.csv', 'a') as f:
        writer = csv.writer(f)
        row = [board, player1, player2, p1_wins, p2_wins, draws, time]
        writer.writerow(row)

if __name__ == '__main__':
    main()


