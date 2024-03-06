# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        foodList = newFood.asList()
        foodDistance = []

        ghostPositions = successorGameState.getGhostPositions()
        ghostDistance = []

        for food in foodList:
            foodDistance.append(manhattanDistance(food, newPos))
        for ghost in ghostPositions:
            ghostDistance.append(manhattanDistance(ghost, newPos))

        if currentGameState.getPacmanPosition() == newPos:
            return -(float("inf"))

        for dist in ghostDistance:
            if dist < 2:
                return -(float("inf"))

        if len(foodDistance) == 0:
            return float("inf")

        return 1000 / sum(foodDistance) + 10000 / len(foodDistance)


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        # left this line in again and wasted time >:(
        # util.raiseNotDefined()

        def max_value(game_State, depth):
            # getting legal actions for the current player (0)
            legalActions = game_State.getLegalActions(0)

            # Checking if the game is over, reached a terminal depth, or if the player won/lost
            if len(legalActions) == 0 or game_State.isWin() or game_State.isLose() or depth == self.depth:
                # Returning the evaluation score but no action
                return self.evaluationFunction(game_State), None

            # We're trying to max so we start at -inf so we can increase
            best_value = -(float("inf"))
            best_action = None

            # Iterating through every available action
            for action in legalActions:
                # Getting the successor state after taking the action we're currently on
                successorValue = min_value(game_State.generateSuccessor(0, action), 1, depth)
                successorValue = successorValue[0]
                # if successors value is greater, we replace the best value since we want to increase
                if successorValue > best_value:
                    best_value, best_action = successorValue, action

            return best_value, best_action

        def min_value(game_State, agentID, depth):
            # Get legal actions for the current agent
            legalActions = game_State.getLegalActions(agentID)

            # Returning the evaluation score and no action, same as before
            if len(legalActions) == 0:
                return self.evaluationFunction(game_State), None
            # Now we want to decrease, so we start from inf and lower the value
            best_value = float("inf")
            best_action = None
            for action in legalActions:
                # if we're dealing with the last agent, we call max_value for the next depth
                if agentID == game_State.getNumAgents() - 1:
                    successorValue = max_value(game_State.generateSuccessor(agentID, action), depth + 1)
                else:
                    # otherwise we call min_value for the next agent
                    successorValue = min_value(game_State.generateSuccessor(agentID, action), agentID + 1, depth)
                successorValue = successorValue[0]
                # If the successor is lower we replace the best value b/c we want to decrease
                if successorValue < best_value:
                    best_value, best_action = successorValue, action
            return best_value, best_action

        # Start from the initial game state
        max_value = max_value(gameState, 0)[1]
        # return the best value we find
        return max_value


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        # util.raiseNotDefined()
        def max_value(gameState, depth, a, b):
            Actions = gameState.getLegalActions(0)  # Get actions of pacman
            if len(Actions) == 0 or gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState), None

            low = -(float("inf"))
            Act = None
            for action in Actions:
                successorValue = min_value(gameState.generateSuccessor(0, action), 1, depth, a, b)
                successorValue = successorValue[0]
                if low < successorValue:
                    low, Act = successorValue, action
                if low > b:
                    return low, Act
                a = max(a, low)
            return low, Act

        def min_value(gameState, agentID, depth, a, b):
            Actions = gameState.getLegalActions(agentID)  # Get the actions of the ghost
            if len(Actions) == 0:
                return self.evaluationFunction(gameState), None
            high = float("inf")
            Act = None
            for action in Actions:
                if agentID == gameState.getNumAgents() - 1:
                    successorValue = max_value(gameState.generateSuccessor(agentID, action), depth + 1, a, b)
                else:
                    successorValue = min_value(gameState.generateSuccessor(agentID, action), agentID + 1, depth, a, b)
                successorValue = successorValue[0]
                if successorValue < high:
                    high, Act = successorValue, action

                if high < a:
                    return high, Act
                b = min(b, high)

            return high, Act

        a = -(float("inf"))
        b = float("inf")
        max_value = max_value(gameState, 0, a, b)[1]
        return max_value


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        # util.raiseNotDefined()

        def max_value(game_State, depth):
            # Get legal actions for the current player (0)
            legalActions = game_State.getLegalActions(0)

            # Returning the eval function value if there aren't legal actions, someone won, or we're at max depth
            if not legalActions or game_State.isWin() or game_State.isLose() or depth == self.depth:
                return self.evaluationFunction(game_State), None

            # Initialize the maximum value to negative infinity and the selected action to None
            max_val = -(float("inf"))
            best_action = None

            # iterate through legal actions
            for action in legalActions:
                # Get the expected value (probability-weighted average) for this action
                exp_val = exp_value(game_State.generateSuccessor(0, action), 1, depth)
                exp_val = exp_val[0]

                # Update the maximum value and selected action if a higher value is found
                if max_val < exp_val:
                    max_val, best_action = exp_val, action

            return max_val, best_action

        def exp_value(game_State, agentID, depth):
            # get legal actions for current player
            legal_actions = game_State.getLegalActions(agentID)

            # if there are no legal actions, return the evaluation function value
            if not legal_actions:
                return self.evaluationFunction(game_State), None

            total_probability = 0
            best_action = None

            # iterating through every legal action
            for action in legal_actions:
                if agentID == game_State.getNumAgents() - 1:
                    # If the current agent is the last agent, calculate the maximum value
                    succ_value = max_value(game_State.generateSuccessor(agentID, action), depth + 1)
                else:
                    # Otherwise, calculate the expected value recursively for the next agent
                    succ_value = exp_value(game_State.generateSuccessor(agentID, action), agentID + 1, depth)

                succ_value = succ_value[0]

                # Calculate the probability of this action and add it to the total
                action_probability = succ_value / len(legal_actions)
                total_probability += action_probability

            return total_probability, best_action

        # Start the max_value function with the initial game state and depth 0, then return the selected action
        selected_action = max_value(gameState, 0)[1]

        return selected_action


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()

    from util import manhattanDistance

    # Storing the current values we need
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    # the manhattan distance to each food from the current pos
    foodList = newFood.asList()
    foodDistance = [0]
    # iterating foods
    for pos in foodList:
        foodDistance.append(manhattanDistance(newPos, pos))

    # the manhattan distance to each ghost from the current pos
    ghostPos = []
    for ghost in newGhostStates:
        ghostPos.append(ghost.getPosition())
    ghostDistance = [0]
    for pos in ghostPos:
        ghostDistance.append(manhattanDistance(newPos, pos))

    numBigFood = len(currentGameState.getCapsules())

    # Defining/storing relevant distances for Pacman to determine priorities
    score = 0
    noEmptySpaces = len(newFood.asList(False))
    totalScaredTime = sum(newScaredTimes)
    totalGhostDistance = sum(ghostDistance)
    foodDist = 0
    if sum(foodDistance) > 0:
        foodDist = 1.0 / sum(foodDistance)

    score += currentGameState.getScore() + foodDist + noEmptySpaces

    if totalScaredTime > 0:
        score += totalScaredTime + (-1 * numBigFood) + (-1 * totalGhostDistance)
    else:
        score += totalGhostDistance + numBigFood
    return score


# Abbreviation
better = betterEvaluationFunction
