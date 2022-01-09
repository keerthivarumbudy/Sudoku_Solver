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
        N = game_state.board.N
        legal_moves = []
        for i in range(N):
            legal_moves.append([])
            for j in range(N):
                legal_moves.__getitem__(i).append(set())
                if game_state.board.get(i, j) == SudokuBoard.empty:
                    for value in range(1, N + 1):
                        if not TabooMove(i, j, value) in game_state.taboo_moves and self.is_legal(game_state, i, j,
                                                                                                  value):
                            legal_moves.__getitem__(i).__getitem__(j).add(value)

        forced_moves = self.check_forced_move(legal_moves)
        old_forced_moves = []

        counter = 0
        while len(forced_moves) > len(old_forced_moves) and counter < 2:
            # print("New: ", len(forced_moves), "Old: ", len(old_forced_moves), "Diff: ", len(forced_moves) - len(old_forced_moves))
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

    def evaluation_function(self, game_state: GameState, moves):
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

    def compute_best_move(self, game_state: GameState) -> None:

        # Create a root for monte_carlo_tree
        [self.player_number, self.opponent_number] = (1, 2) if len(game_state.moves) % 2 == 0 else (2, 1)
        root = Node(game_state, curr_player=self.opponent_number)
        root.n = 1
        root.p = game_state.scores[self.player_number - 1] - game_state.scores[self.opponent_number - 1]

        # Add children nodes of root to the tree
        initial_legal_moves = self.find_legal_moves(game_state)
        root.legal_moves = initial_legal_moves
        new_moves = self.l2a(initial_legal_moves)
        self.propose_move(new_moves[0])
        for i in range(len(new_moves)):
            move = new_moves[i]
            new_game_state = copy.deepcopy(root.game_state)
            new_game_state.board.put(move.i, move.j, move.value)
            new_game_state.scores[self.player_number - 1] += self.compute_move_score(root.game_state.board, move)
            new_game_state.moves.append(move)
            new_node = Node(game_state=new_game_state, move=move, parent=root, curr_player=self.player_number)
            root.children[i] = new_node

        # Do searching till run out of time
        count = 0
        while count < 20:
            count += 1
            print("search:"+str(count))
            self.monte_carlo_tree_search(root)

    def monte_carlo_tree_search(self, root):
        # find the leaf node we need to simulate
        leaf = self.traverse(root)
        print('leaf:'+str(leaf.move))

        # simulate the rest game
        simulation_result = self.simulate(leaf)
        print('simulation_result:' + str(simulation_result))

        # back propagate according to result of simulation and update the proposed move
        self.backPropagate(leaf, simulation_result)

    def traverse(self, node):
        """
            find the node we need to simulate
        """
        node = node
        # keep going deeper to a leaf
        while node.children:
            node = node.best_utc()
            print('pick:'+str(node.move))

        # if the leaf is unvisited, return the leaf
        if node.n == 0:
            return node

        # if the leaf is visited, create children nodes for the leaf and add them to the tree
        lm = copy.deepcopy(node.parent.legal_moves)
        new_moves = self.update_lm(node.game_state, lm, move=node.move)
        node.legal_moves = new_moves
        moves = self.l2a(new_moves)
        if moves:
            for i in range(len(moves)):
                move = moves[i]
                new_game_state = copy.deepcopy(node.game_state)
                new_game_state.board.put(move.i, move.j, move.value)
                if node.curr_player == self.player_number:
                    new_game_state.scores[self.opponent_number - 1] += self.compute_move_score(node.game_state.board,
                                                                                               move)
                    next_player = self.opponent_number
                else:
                    new_game_state.scores[self.player_number - 1] += self.compute_move_score(node.game_state.board,
                                                                                             move)
                    next_player = self.player_number
                new_game_state.moves.append(move)
                child = Node(game_state=new_game_state, move=move, parent=node, curr_player=next_player)
                node.children[i] = child
                # return the first child of the leaf
            return node.children[0]
        else:
            return node

    def simulate(self, node):
        """
            simulation result of input node
        """
        # simulate the game
        lm = copy.deepcopy(node.parent.legal_moves)
        new_moves = self.update_lm(node.parent.game_state, lm, move=node.move)
        moves = self.l2a(new_moves)
        return self.evaluation_function(node.game_state, moves)

    def backPropagate(self, node, result):
        # if reach root, propose a move
        if node.parent is None:
            move = node.best_n().move
            self.propose_move(move)
            print('propose:'+str(move))
            return

        # update state of node in the path
        node.n += 1
        c = 1 if node.curr_player == self.player_number else -1
        node.p += c * result
        self.backPropagate(node.parent, result)


class Node:
    """
    Class of Monte Carlo Tree Searching
    """

    def __init__(self, game_state: GameState, move=None, parent=None, curr_player=0):
        self.game_state = game_state
        self.move = move
        self.parent = parent
        self.children = {}
        self.p = 0
        self.n = 0
        self.curr_player = curr_player
        self.legal_moves = None

    def best_n(self):
        index = 0
        max_n = 0
        for i in range(len(self.children)):
            curr_n = self.children.get(i).n
            if curr_n > max_n:
                max_n = curr_n
                index = i
        return self.children.get(index)

    def best_utc(self):
        index = 0
        best_utc = -math.inf
        for i in range(len(self.children)):
            curr_utc = self.children.get(i).utc()
            # print('move:'+str(self.children.get(i).move))
            # print('move_utc:' + str(self.children.get(i).utc()))
            if curr_utc > best_utc:
                best_utc = curr_utc
                index = i
        return self.children.get(index)

    def utc(self):
        if self.n == 0:
            return math.inf
        else:
            c = 2
            score = self.p / self.n + c * ((math.log(self.parent.n) / self.n) ** 0.5)
            return score
