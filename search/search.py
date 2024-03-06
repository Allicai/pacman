# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()

    from util import Stack

    fringe = Stack()
    visited = set()
    startState = problem.getStartState()
    fringe.push((startState, []))
    # the tuple is formatted like this: (state, path)

    # check if the fringe is empty
    while not fringe.isEmpty():
        # pop if not
        currentState, actions = fringe.pop()
        # if we haven't visited the current state before
        if currentState not in visited:
            # add it to the states we have visited
            visited.add(currentState)

            if problem.isGoalState(currentState):
                return actions

            successors = problem.getSuccessors(currentState)
            for nextState, action, _ in successors:
                if nextState not in visited:
                    nextActions = actions + [action]
                    fringe.push((nextState, nextActions))

    return []  # Return an empty list if no path is found


def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()

    # had to redo my style for this question because implementing it how I implemented problem 1
    # had me stuck on how to define a starting state, was getting errors I couldn't make sense of
    from util import Queue

    # Create an empty Queue to serve as the fringe.
    fringe = Queue()

    visited_list = []  # List to keep track of visited states.
    path = []  # List to store the current path.
    # set a cost of 0
    action_cost = 0

    start_position = problem.getStartState()

    # Push the start position to the fringe.
    fringe.push((start_position, path, action_cost))

    # make sure fringe isnt empty
    while not fringe.isEmpty():
        # pop nodes
        current_node = fringe.pop()
        # take the pos and path
        position = current_node[0]
        path = current_node[1]

        # check if we've visited the current pos before, otherwise add it to our list
        if position not in visited_list:
            visited_list.append(position)

        # check if we start in the goal state
        if problem.isGoalState(position):
            return path  # return path if we reach the goal

        # get successors
        succ = problem.getSuccessors(position)

        # add successors to the fringe if they aren't already in it or havent already been visited
        for item in succ:
            # successors store the position, action, and cost
            successor_position = item[0]
            successor_action = item[1]
            successor_cost = item[2]

            # Check if the successor is neither visited nor in the fringe.
            if successor_position not in visited_list and not any(
                    node[0] == successor_position for node in fringe.list):
                new_path = path + [successor_action]
                fringe.push((successor_position, new_path, successor_cost))


def uniformCostSearch(problem: SearchProblem):
    """Search the node of the least total cost first."""
    "*** YOUR CODE HERE ***"

    # literally implemented my method without commenting the line below out and spent so long on this :/
    # util.raiseNotDefined()

    # UCS should use a PriorityQueue instead of a normal queue, since we need to prioritize by cost and can't just do
    # FIFO principles for this algorithm
    from util import PriorityQueue

    # Initialize the priority queue for storing nodes to explore.
    # We use a priority queue because we need to prioritize nodes by cost.
    fringe = PriorityQueue()

    # Start with the initial state and a cost of 0.
    fringe.push(problem.getStartState(), 0)

    # Initialize variables to store the final path, visited nodes, and temporary paths.
    final_path = []  # This will store the final actions/directions.
    visited_states = []  # This will store visited states.
    temp_path = []  # This is used for constructing the path incrementally.

    # init a priority queue
    path_to_current = PriorityQueue()

    # pop the queue to get the current state
    current_state = fringe.pop()

    # search until we achieve the goal state, serves as an initial goal state check as well
    while not problem.isGoalState(current_state):
        # check if we've been to the current state
        if current_state not in visited_states:
            visited_states.append(current_state)  # mark as visited

            # get the successors
            successors = problem.getSuccessors(current_state)

            for succ, action, step_cost in successors:
                # temp path construction
                temp_path = final_path + [action]

                # find the cost to reach the successor
                cost_to_reach = problem.getCostOfActions(temp_path)

                # check if we've visited the succ
                if succ not in visited_states:
                    # add succ node and its cost to queue
                    fringe.push(succ, cost_to_reach)

                    # add the temporary path to the current node to the path_to_current priority queue
                    path_to_current.push(temp_path, cost_to_reach)

        # pop the queue, move to the node with the lowest priority value / highest actual priority
        current_state = fringe.pop()

        # update the final path with path to current state/node
        final_path = path_to_current.pop()

    # return path at goal state
    return final_path


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    # util.raiseNotDefined()

    # A* also needs a PriorityQueue for keeping track of total cost up to the current state
    # as well as the estimated cost to reach the goal
    from util import PriorityQueue
    fringe = PriorityQueue()  # Fringe to manage which states to expand
    fringe.push(problem.getStartState(), 0)
    currState = fringe.pop()
    visited = []  # List to check whether state has already been visited
    tempPath = []  # Temp variable to get intermediate paths
    path = []  # List to store final sequence of directions
    pathToCurrent = PriorityQueue()  # Queue to store direction to children (currState and pathToCurrent go hand in hand)
    while not problem.isGoalState(currState):
        # if current state hasn't been visited yet
        if currState not in visited:
            visited.append(currState)
            successors = problem.getSuccessors(currState)
            for succ, direction, cost in successors:
                tempPath = path + [direction]
                costToGo = problem.getCostOfActions(tempPath) + heuristic(succ, problem)
                # heuristic search is fallible
                if succ not in visited:
                    fringe.push(succ, costToGo)
                    pathToCurrent.push(tempPath, costToGo)
        currState = fringe.pop()
        path = pathToCurrent.pop()
    return path


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
