import random
from math import log, sqrt
from typing import List
from time import time
from isolation import Isolation
from isolation.isolation import Action
from sample_players import DataPlayer

TIME_LIMIT_IN_SECONDS = 0.145
VAL_MUL_UCB1 = 2
TH_NUM_PLAYS = 20


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

    def get_action(self, state: Isolation):
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
        root_node = self._get_node_mcts(state, tree)
        mc_searcher = MCSearcher(tree, root_node)

        start_time = time()
        while time() - start_time < TIME_LIMIT_IN_SECONDS:
            mc_searcher.iterate()

        self._put_action_in_queue(root_node)

    def _put_action_in_queue(self, root_node: 'NodeMCTS'):
        children = root_node.children

        if children:
            action = max(children, key=lambda x: x.plays).action
        else:
            action = random.choice(root_node.state.actions())

        self.queue.put(action)

    def _get_node_mcts(self, state: Isolation, tree: dict) -> 'NodeMCTS':
        if state in tree.keys():
            node_mcts = tree[state]
        else:
            node_mcts = self._create_root(state, tree)

        return node_mcts

    # noinspection PyMethodMayBeStatic
    def _create_root(self, state: Isolation, tree: dict) -> 'NodeMCTS':
        node_mcts = NodeMCTS(state)
        tree[state] = node_mcts

        return node_mcts


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


class MCSearcher:
    __slots__ = ('_tree', '_root_node')

    def __init__(self, tree: dict, root_node: NodeMCTS):
        self._tree = tree
        self._root_node = root_node

    def iterate(self):
        leaf_node = self._get_leaf_node(self._root_node)
        expanded_node = self._expand_children(leaf_node)
        self._pass_val_to_parent_nodes(self._get_utility_from_simulation(expanded_node.state, expanded_node.state.player()), expanded_node)

    def _get_leaf_node(self, node: NodeMCTS) -> NodeMCTS:
        while True:
            children = node.children

            if children:
                for child in children:
                    if child.plays == 0:
                        return child

                if node.plays < TH_NUM_PLAYS:
                    node = random.choice(children)
                else:
                    node = self._ucb1(children)
            else:
                return node

    def _ucb1(self, children: List[NodeMCTS]) -> NodeMCTS:
        list_values = []
        log_children0_parent_plays = log(children[0].parent.plays)

        for child in children:
            value = child.wins / child.plays + VAL_MUL_UCB1 * sqrt(log_children0_parent_plays / child.plays)
            list_values.append((value, child))

        return max(list_values, key=lambda x: x[0])[1]

    def _expand_children(self, leaf_node: NodeMCTS) -> NodeMCTS:
        return leaf_node if leaf_node.state.terminal_test() else random.choice(self._create_children(leaf_node))

    def _create_children(self, parent_node: NodeMCTS) -> List[NodeMCTS]:
        for action in parent_node.state.actions():
            state = parent_node.state.result(action)
            self._tree[state] = parent_node.create_child(action, state)

        return parent_node.children

    # noinspection PyMethodMayBeStatic
    def _get_utility_from_simulation(self, state: Isolation, leaf_player_id: int) -> float:
        while True:
            if state.terminal_test():
                return state.utility(leaf_player_id)

            state = state.result(random.choice(state.actions()))

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
