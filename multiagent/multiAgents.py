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
from game import Agent
import random
import util


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(
            len(scores)) if scores[index] == bestScore]
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)

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

        return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState):
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

    def isEnd(self, state, depth):
        return state.isWin() or state.isLose() or depth == self.depth * state.getNumAgents()

    def getAgent(self, state, depth):
        return depth % state.getNumAgents()


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 1)
    """

    def ghostFunc(self, vals):
        return min(vals)

    def getAction(self, gameState):
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

        return self.recurse(gameState, 0)[1]

    def recurse(self, state, depth):
        if self.isEnd(state, depth):
            return self.evaluationFunction(state), Directions.STOP

        agent = self.getAgent(state, depth)
        actions = state.getLegalActions(agent)
        vals = [self.recurse(state.generateSuccessor(agent, a), depth + 1)[0]
                for a in actions]
        val = (max if agent == 0 else self.ghostFunc)(vals)
        act = actions[vals.index(val)] if agent == 0 else Directions.STOP

        return val, act


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        return self.recurseStates(gameState, 0, float('-inf'), float('inf'))[1]

    def recurseStates(self, state, depth, alpha, beta):
        if self.isEnd(state, depth):
            return self.evaluationFunction(state), Directions.STOP

        return (self.recurseBeta if self.getAgent(state, depth) == 0 else
                self.recurseAlpha)(state, depth, alpha, beta)

    def recurseBeta(self, state, depth, alpha, beta):
        curr = float('-inf')
        agent = self.getAgent(state, depth)
        action = Directions.STOP
        for a in state.getLegalActions(agent):
            val, _ = self.recurseStates(state.generateSuccessor(
                agent, a), depth+1, alpha, beta)
            if val > curr:
                curr = val
                action = a
            if curr > beta:  # Exit early, i.e. prune
                return (curr, action)
            if curr > alpha:
                alpha = curr
        return curr, action

    def recurseAlpha(self, state, depth, alpha, beta):
        curr = float('inf')
        agent = self.getAgent(state, depth)
        action = Directions.STOP
        for a in state.getLegalActions(agent):
            succ = state.generateSuccessor(agent, a)
            val, _ = self.recurseStates(succ, depth+1, alpha, beta)
            if val < curr:
                curr = val
                action = a
            if curr < alpha:
                return (curr, action)
            if curr < beta:
                beta = curr
        return curr, action


class ExpectimaxAgent(MinimaxAgent):
    """
      Your expectimax agent (question 3)
    """

    def ghostFunc(self, vals):
        return sum(vals) / len(vals)

    """
    `getAction` is inherited from MinimaxAgent
    """


def betterEvaluationFunction(currentGameState, init=True):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 4).
    """
    original = currentGameState.getScore()
    pos = currentGameState.getPacmanPosition()
    foods = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()

    """
    - Use randint to break ties when min(foodDists) is same at different states.
    - `time` - 2 > 0 when ghosts are going to be scared for > 2 second.
    - Otherwise it may be sensible to run away from ghost
    """
    ghostScaredTimesAndDists = [(g.scaredTimer, manhattanDistance(pos, g.getPosition()))
                                for g in ghostStates]
    foodDists = [manhattanDistance(pos, food) for food in foods.asList()]

    def calcGhostScore(tup):
        time, dist = tup
        return 2 * (time - 2) / (dist / 2 + 0.1)

    foodScore = min(foodDists, default=0)
    ghostScore = sum(map(calcGhostScore, ghostScaredTimesAndDists))

    return original - foodScore + ghostScore + random.randint(0, 1)




# Abbreviation
better = betterEvaluationFunction
