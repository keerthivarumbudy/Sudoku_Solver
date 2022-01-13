#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

import copy
import math

import competitive_sudoku.sudokuai
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove


class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    Sudoku AI that computes a move for a given sudoku configuration.
    """
    player_number = 1           # the order of our AI
    opponent_number = 2         # the order of opponent
    found_taboo = False         # if we find a taboo move
    exp_taboo = Move(0, 0, 0)   # the taboo move we can use

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
            eval_value = diff_score + 0.5 * curr_player * heuristic_point

        return eval_value

    def is_board_full(self, game_state: GameState):
        """
            Function to check if a board is completely filled
        """
        N = game_state.board.N
        length_move = len(game_state.moves)
        if length_move < N*N:
            return False
        else:
            return True

    def compute_best_move(self, game_state: GameState) -> None:

        # Create the root for monte carlo tree search
        [self.player_number, self.opponent_number] = (1, 2) if len(game_state.moves) % 2 == 0 else (2, 1)
        root = Node(game_state, curr_player=self.opponent_number)
        root.n = 1
        root.p = game_state.scores[self.player_number - 1] - game_state.scores[self.opponent_number - 1]

        # Add children nodes of root to the tree
        initial_legal_moves = self.find_legal_moves(game_state)
        root.legal_moves = initial_legal_moves
        root.num_lm = len(self.l2a(initial_legal_moves))

        # Do monte carlo tree search till run out of time
        while True:
            self.monte_carlo_tree_search(root)

    def monte_carlo_tree_search(self, root):

        # record the current difference of points
        real_diff = root.game_state.scores[self.player_number - 1] = root.game_state.scores[self.opponent_number - 1]
        # find the leaf node we need to simulate
        leaf = self.traverse(root)
        # simulate the rest game
        simulation_result = self.simulate(leaf, real_diff)
        # back propagate according to result of simulation and update the proposed move
        self.backPropagate(leaf, simulation_result)

    def traverse(self, node):
        """
            find the node we need to simulate
        """
        # keep going deeper to a leaf
        node = node
        while node.children and node.num_lm == len(node.children):
            node = node.best_utc()

        # if the leaf is unvisited, return it
        if node.n == 0:
            return node

        # if the leaf is visited, create children nodes for the leaf and add them to the tree:
        # find legal moves if we have not done that for this node yet.
        if node.legal_moves is None:  # if node is root
            lm = copy.deepcopy(node.parent.legal_moves)
            new_moves = self.update_lm(node.game_state, lm, move=node.move)
            node.legal_moves = new_moves
            node.num_lm = len(self.l2a(new_moves))
        else:
            new_moves = node.legal_moves

        moves = self.l2a(new_moves)

        # create a child node for next legal move and return it
        if moves:
            # find the index of next move we need to explore
            i = len(node.children)
            move = moves[i]
            # update game_state
            new_game_state = copy.deepcopy(node.game_state)
            new_game_state.board.put(move.i, move.j, move.value)
            new_game_state.moves.append(move)
            if node.curr_player == self.player_number:
                new_game_state.scores[self.opponent_number - 1] += self.compute_move_score(node.game_state.board,
                                                                                           move)
                next_player = self.opponent_number
            else:
                new_game_state.scores[self.player_number - 1] += self.compute_move_score(node.game_state.board,
                                                                                         move)
                next_player = self.player_number
            #
            child = Node(game_state=new_game_state, move=move, parent=node, curr_player=next_player)
            node.children[i] = child
            return node.children[i]

        else:  # if the leaf node have no child, return the node itself
            return node

    def simulate(self, node, real_diff):
        """
            simulation result of input node
        """
        # update legal moves one more time
        lm = copy.deepcopy(node.parent.legal_moves)
        new_moves = self.update_lm(node.parent.game_state, lm, move=node.move)
        moves = self.l2a(new_moves)

        # if there is taboo move in Search Tree return the difference of points of root state
        # else simulate result by eval_func
        if len(moves) == 0 and self.is_board_full(node.game_state):
            score = real_diff
        else:
            score = self.evaluation_function(node.game_state, moves)

        return score

    def backPropagate(self, node, result):
        # if reach root, propose a move
        if node.parent is None:
            move, max_value = node.best_pn()
            # if our AI won't play the last move and we have no good move in this turn,
            # then propose taboo move if we find one.
            if max_value <= 0\
                    and (math.sqrt(node.game_state.board.N) - len(node.game_state.moves)) % 2 == 0\
                    and self.found_taboo:
                self.propose_move(self.exp_taboo)
            else:
                self.propose_move(move)
            return

        # update state of nodes in the path from the leaf to root
        node.n += 1
        c = 1 if node.curr_player == self.player_number else -1
        node.p += c * result
        self.backPropagate(node.parent, result)


class Node:
    """
        Monte Carlo Tree Searching
    """

    def __init__(self, game_state: GameState, move=None, parent=None, curr_player=0):
        self.game_state = game_state        # store the game_state after this move
        self.move = move                    # store the move made in this node
        self.parent = parent                # point to parent node
        self.children = {}                  # store all children nodes
        self.p = 0                          # a value used by MCTS: the result accumulated during visits
        self.n = 0                          # a value used by MCTS: the number of times the node be visited
        self.curr_player = curr_player      # which player's turn
        self.legal_moves = None             # store all legal move
        self.num_lm = 0                     # number of legal moves

    def best_n(self):
        """
            Choose the move of child node with max n
        """
        index = 0
        max_n = 0
        for i in range(len(self.children)):
            curr_n = self.children.get(i).n
            if curr_n > max_n:
                max_n = curr_n
                index = i
        return self.children.get(index).move, max_n

    def best_pn(self):
        """
            Choose the move of child node with max p/n
        """
        index = 0
        max_pn = -math.inf
        curr_pn = -math.inf
        for i in range(len(self.children)):
            n = self.children.get(i).n
            p = self.children.get(i).p
            if n > 0:
                curr_pn = p / n
            if curr_pn > max_pn:
                max_pn = curr_pn
                index = i
        return self.children.get(index).move, max_pn

    def best_utc(self):
        """
            Find best child node with max utc
        """
        index = 0
        best_utc = -math.inf
        for i in range(len(self.children)):
            curr_utc = self.children.get(i).utc()
            if curr_utc > best_utc:
                best_utc = curr_utc
                index = i
        return self.children.get(index)

    def utc(self):
        """
            Compute utc value for MCTS
        """
        if self.n == 0:
            return math.inf
        else:
            c = 2
            score = self.p / self.n + c * ((math.log(self.parent.n) / self.n) ** 0.5)
            return score
