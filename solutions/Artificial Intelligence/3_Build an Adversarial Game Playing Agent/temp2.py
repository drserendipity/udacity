import random
from math import log, sqrt
from typing import List, Dict
from time import time
from isolation import Isolation
from isolation.isolation import Action
from sample_players import DataPlayer

# TREE_PICKLE = 'data.pickle'
TIME_LIMIT_IN_SECONDS = 0.145
VAL_MUL_UCB1 = 0.7
TH_NUM_PLAYS = 10

class NodeMCTS:
    __slots__ = ('state', 'action', 'parent', 'children', 'plays', 'wins')

    def __init__(self, state: Isolation, action: Action = None, parent: 'NodeMCTS' = None):
        self.state = state
        self.action = action
        self.parent = parent
        self.children = []
        self.plays = 0
        self.wins = 0.0

    def create_child(self, action: Action, state: Isolation) -> 'NodeMCTS':
        child = NodeMCTS(state, action=action, parent=self)
        self.children.append(child)
        return child

    # @classmethod
    # def create_state_tree(cls, root_node: 'NodeMCTS', turns: int = 99999):
    #     tree = {}
    #     starting_depth = root_node.state.ply_count
    #     max_depth = starting_depth + turns * 2
    #     stack = [root_node]
    #     while stack:
    #         node = stack.pop()
    #         tree[node.state] = node
    #         if node.state.ply_count <= max_depth:
    #             stack.extend(node.children)
    #     return tree


class MonteCarloSearcher:
    __slots__ = ('_tree', '_root_node')

    def __init__(self, tree: dict, root_node: NodeMCTS):
        self._tree = tree
        self._root_node = root_node

    # def get_tree(self):
    #     return self._tree

    def iterate_once(self):
        leaf_node = self._selection(self._root_node)
        leaf_or_child = self._expansion(leaf_node)
        utility = self._simulation(leaf_or_child.state, leaf_or_child.state.player())
        self._backpropagation(utility, leaf_or_child)

    def _selection(self, node: NodeMCTS) -> NodeMCTS:
        while True:
            children = node.children
            if children:
                assert len(children) == len(node.state.actions())
                for child in children:
                    if child.plays == 0: return child

                if node.plays < TH_NUM_PLAYS:
                    node = random.choice(children)
                else:
                    node = self._ucb1(children)
            else:
                return node

    def _ucb1(self, children: List[NodeMCTS]):
        log_parent_plays = log(children[0].parent.plays)
        values = []
        for child in children:
            v = child.wins / child.plays + VAL_MUL_UCB1 * sqrt(log_parent_plays / child.plays)
            values.append((v, child))
        return max(values, key=lambda x: x[0])[1]

    def _expansion(self, leaf_node: NodeMCTS) -> NodeMCTS:
        if leaf_node.state.terminal_test(): return leaf_node
        children = self._create_children(leaf_node)
        return random.choice(children)

    def _create_children(self, parent_node: NodeMCTS):
        for action in parent_node.state.actions():
            child_state = parent_node.state.result(action)
            child_node = parent_node.create_child(action, child_state)
            self._tree[child_state] = child_node
        return parent_node.children

    def _simulation(self, state: Isolation, leaf_player_id: int) -> float:
        while True:
            if state.terminal_test(): return state.utility(leaf_player_id)
            state = state.result(random.choice(state.actions()))

    def _backpropagation(self, utility: float, node: NodeMCTS):
        leaf_player = node.state.player()
        while node:
            node.plays += 1
            if utility == 0:
                node.wins += 0.5
            else:
                player = node.state.player()
                if (utility < 0 and player == leaf_player) or (utility > 0 and player != leaf_player):
                    node.wins += 1

            if node == self._root_node:
                return
            else:
                node = node.parent


class CustomPlayer(DataPlayer):
    """ Implement your own agent to play knight's Isolation
    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding *named parameters
    with default values*, but the function MUST remain compatible with the
    default
    **********************************************************************
    NOTES:
    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.
    - You can pass state forward to your agent on the next turn by assigning
      any pickleable object to the self.context attribute.
    **********************************************************************
    """

    def get_action(self, state: Isolation):
        tree = {}

        root_node_for_turn = self._get_node_mcts(state, tree)
        monte_carlo_searcher = MonteCarloSearcher(tree, root_node_for_turn)

        start_time = time()
        while time() - start_time < TIME_LIMIT_IN_SECONDS:
            monte_carlo_searcher.iterate_once()

        self._select_action(root_node_for_turn)

    def _select_action(self, root_node_for_turn: NodeMCTS):
        action = self._most_played_node(root_node_for_turn)
        self.queue.put(action)

    def _most_played_node(self, root_node_for_turn: NodeMCTS) -> Action:
        children = root_node_for_turn.children
        if children:
            action = max(children, key=lambda e: e.plays).action
        else:
            action = random.choice(root_node_for_turn.state.actions())
        return action

    def _get_node_mcts(self, state: Isolation, tree: dict):
        if state in tree.keys():
            node_mcts = tree[state]
        else:
            node_mcts = self._create_root(state, tree)
        return node_mcts

    def _create_root(self, state: Isolation, tree: dict):
        node_mcts = NodeMCTS(state)
        tree[state] = node_mcts
        return node_mcts

    # def minimax_iterative_deepening(self, state: Isolation):
    #     alpha, beta = float("-inf"), float("inf")
    #     depth = 1
    #     while True:
    #         self.queue.put(self._minimax_with_alpha_beta_pruning(state, depth, alpha, beta))
    #         depth += 1
    #
    # def _minimax_with_alpha_beta_pruning(self, state: Isolation, depth: int, alpha: float, beta: float) -> Action:
    #
    #     def min_value(state: Isolation, depth: int, alpha: float, beta: float):
    #         if state.terminal_test(): return state.utility(self.player_id)
    #         if depth <= 0: return self._evaluate(state)
    #         value = float("inf")
    #         for action in state.actions():
    #             value = min(value, max_value(state.result(action), depth - 1, alpha, beta))
    #
    #             if value <= alpha: break
    #             beta = min(beta, value)
    #
    #         return value
    #
    #     def max_value(state: Isolation, depth: int, alpha: float, beta: float):
    #         if state.terminal_test(): return state.utility(self.player_id)
    #         if depth <= 0: return self._evaluate(state)
    #         value = float("-inf")
    #         for action in state.actions():
    #             value = max(value, min_value(state.result(action), depth - 1, alpha, beta))
    #
    #             if value >= beta: break
    #             alpha = max(alpha, value)
    #
    #         return value
    #
    #     return max(state.actions(), key=lambda x: min_value(state.result(x), depth - 1, alpha, beta))
    #
    # def _evaluate(self, state: Isolation):
    #     return self._score(state)
    #
    # def _score(self, state: Isolation):
    #     own_loc = state.locs[self.player_id]
    #     opp_loc = state.locs[1 - self.player_id]
    #     own_liberties = state.liberties(own_loc)
    #     opp_liberties = state.liberties(opp_loc)
    #     return len(own_liberties) - len(opp_liberties)