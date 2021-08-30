import random
import numpy as np
from math import log, sqrt
from time import time
from isolation import Isolation
from isolation.isolation import Action
from sample_players import DataPlayer

TIME_LIMIT_IN_SEC = 0.145
TH_NUM_PLAYS = 10
VAL_MUL_UCB1 = 0.7


class CustomPlayer(DataPlayer):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding named parameters
    with default values, but the function MUST remain compatible with the
    default interface.

    **********************************************************************
    NOTES:
    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.

    - You can pass state forward to your agent on the next turn by assigning
      any pickleable object to the self.context attribute.
    **********************************************************************
    """

    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller will be responsible
        for cutting off the function after the search time limit has expired.

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE:
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        # TODO: Replace the example implementation below with your own search
        #       method by combining techniques from lecture
        #
        # EXAMPLE: choose a random move without any search--this function MUST
        #          call self.queue.put(ACTION) at least once before time expires
        #          (the timer is automatically managed for you)
        # self.queue.put(random.choice(state.actions()))
        tree = {}
        root_node_for_turn = self._get_state_node(state, tree)
        monte_carlo_searcher = MonteCarloSearcher(tree, root_node_for_turn)

        start_time = time()
        while time() - start_time < TIME_LIMIT_IN_SEC:
            monte_carlo_searcher.iterate_once()

        self._select_action(root_node_for_turn)

    def _select_action(self, root_node_for_turn):
        action = self._most_played_node(root_node_for_turn)
        self.queue.put(action)

    # noinspection PyMethodMayBeStatic
    def _most_played_node(self, root_node_for_turn) -> Action:
        children = root_node_for_turn.children
        if children:
            action = max(children, key=lambda e: e.plays).action
        else:
            action = random.choice(root_node_for_turn.state.actions())
        return action

    def _get_state_node(self, state, tree):
        if state in tree.keys():
            state_node = tree[state]
        else:
            state_node = self._create_root(state, tree)
        return state_node

    # noinspection PyMethodMayBeStatic
    def _create_root(self, state: Isolation, tree):
        state_node = NodeMCTS(state)
        tree[state] = state_node
        return state_node


class NodeMCTS:
    __slots__ = ('state', 'action', 'parent', 'children', 'plays', 'wins')

    def __init__(self, state: Isolation, action: Action = None, parent: 'NodeMCTS' = None):
        self.state = state
        self.action = action
        self.parent = parent
        self.children = []
        self.plays = 0
        self.wins = 0.0

    def create_child(self, action, state) -> 'NodeMCTS':
        child = NodeMCTS(state, action=action, parent=self)
        self.children.append(child)
        return child


class MonteCarloSearcher:
    __slots__ = ('_tree', '_root_node')

    def __init__(self, tree: dict, root_node: NodeMCTS):
        self._tree = tree
        self._root_node = root_node

    def iterate_once(self):
        leaf_node = select_node(self._root_node)
        leaf_or_child = self._expansion(leaf_node)
        utility = simulate_mc(leaf_or_child.state, leaf_or_child.state.player())
        self._pass_val_to_parent_nodes(utility, leaf_or_child)

    def _expansion(self, leaf_node: NodeMCTS) -> NodeMCTS:
        if leaf_node.state.terminal_test():
            return leaf_node
        else:
            children = self._create_children(leaf_node)
            return random.choice(children)

    def _create_children(self, parent_node: NodeMCTS):
        for action in parent_node.state.actions():
            child_state = parent_node.state.result(action)
            child_node = parent_node.create_child(action, child_state)
            self._tree[child_state] = child_node

        return parent_node.children

    def _pass_val_to_parent_nodes(self, utility: float, node: NodeMCTS):
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


def simulate_mc(state: Isolation, leaf_player_id: int) -> float:
    while True:
        if state.terminal_test():
            return state.utility(leaf_player_id)
        else:
            action = random.choice(state.actions())
            state = state.result(action)


def select_node(node: NodeMCTS) -> NodeMCTS:
    while True:
        children = node.children
        if children:
            # assert len(children) == len(node.state.actions())
            for child in children:
                if child.plays == 0:
                    return child

            if node.plays < TH_NUM_PLAYS:
                node = random.choice(children)
            else:
                node = ucb1(children)
        else:
            return node


def ucb1(children):
    values = []
    log_parent_plays = log(children[0].parent.plays)

    for child in children:
        val = child.wins / child.plays + VAL_MUL_UCB1 * sqrt(log_parent_plays / child.plays)
        values.append((val, child))

    return max(values, key=lambda x: x[0])[1]
