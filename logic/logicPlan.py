# logicPlan.py
# ------------
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
In logicPlan.py, you will implement logic planning methods which are called by
Pacman agents (in logicAgents.py).
"""

import util
import sys

from logic import *
from itertools import combinations, product

pacman_str = 'P'
ghost_pos_str = 'G'
ghost_east_str = 'GE'
pacman_alive_str = 'PA'


class PlanningProblem:
    """
    This class outlines the structure of a planning problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the planning problem.
        """
        util.raiseNotDefined()

    def getGhostStartStates(self):
        """
        Returns a list containing the start state for each ghost.
        Only used in problems that use ghosts (FoodGhostPlanningProblem)
        """
        util.raiseNotDefined()

    def getGoalState(self):
        """
        Returns goal state for problem. Note only defined for problems that have
        a unique goal state such as PositionPlanningProblem
        """
        util.raiseNotDefined()


def tinyMazePlan(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def sentence1():
    """Returns a Expr instance that encodes that the following expressions are all true.

    A or B
    (not A) if and only if ((not B) or C)
    (not A) or (not B) or C
    """
    A = Expr('A')
    B = Expr('B')
    C = Expr('C')
    return conjoin([(A | B), (~A % (~B | C)), disjoin([~A, ~B, C])])


def sentence2():
    """Returns a Expr instance that encodes that the following expressions are all true.

    C if and only if (B or D)
    A implies ((not B) and (not D))
    (not (B and (not C))) implies A
    (not D) implies C
    """
    A = Expr('A')
    B = Expr('B')
    C = Expr('C')
    D = Expr('D')
    return conjoin([(C % (B | D)), A >> (~B & ~D), ~(B & ~C) >> A, ~D >> C])


def sentence3():
    """Using the symbols WumpusAlive[1], WumpusAlive[0], WumpusBorn[0], and WumpusKilled[0],
    created using the PropSymbolExpr constructor, return a PropSymbolExpr
    instance that encodes the following English sentences (in this order):

    The Wumpus is alive at time 1 if and only if the Wumpus was alive at time 0 and it was
    not killed at time 0 or it was not alive and time 0 and it was born at time 0.

    The Wumpus cannot both be alive at time 0 and be born at time 0.

    The Wumpus is born at time 0.
    """
    Alive1 = PropSymbolExpr('WumpusAlive', 1)
    Alive0 = PropSymbolExpr('WumpusAlive', 0)
    Born0 = PropSymbolExpr('WumpusBorn', 0)
    Killed0 = PropSymbolExpr('WumpusKilled', 0)

    return conjoin([Alive1 % ((Alive0 & ~Killed0) | (~Alive0 & Born0)), ~(Alive0 & Born0), Born0])


def findModel(sentence):
    """Given a propositional logic sentence (i.e. a Expr instance), returns a satisfying
    model if one exists. Otherwise, returns False.
    """
    return pycoSAT(to_cnf(sentence))


def atLeastOne(literals):
    """
    Given a list of Expr literals (i.e. in the form A or ~A), return a single
    Expr instance in CNF (conjunctive normal form) that represents the logic
    that at least one of the literals in the list is true.
    >>> A = PropSymbolExpr('A');
    >>> B = PropSymbolExpr('B');
    >>> symbols = [A, B]
    >>> atleast1 = atLeastOne(symbols)
    >>> model1 = {A:False, B:False}
    >>> print pl_true(atleast1,model1)
    False
    >>> model2 = {A:False, B:True}
    >>> print pl_true(atleast1,model2)
    True
    >>> model3 = {A:True, B:True}
    >>> print pl_true(atleast1,model2)
    True
    """
    return disjoin(literals)


def atMostOne(literals):
    """
    Given a list of Expr literals, return a single Expr instance in
    CNF (conjunctive normal form) that represents the logic that at most one of
    the expressions in the list is true.
    """
    return conjoin([disjoin(~l0, ~l1) for l0, l1 in combinations(literals, 2)])


def exactlyOne(literals):
    """
    Given a list of Expr literals, return a single Expr instance in
    CNF (conjunctive normal form)that represents the logic that exactly one of
    the expressions in the list is true.
    """
    return atLeastOne(literals) & atMostOne(literals)


def extractActionSequence(model, actions):
    """
    Convert a model in to an ordered list of actions.
    model: Propositional logic model stored as a dictionary with keys being
    the symbol strings and values being Boolean: True or False
    Example:
    >>> model = {"North[3]":True, "P[3,4,1]":True, "P[3,3,1]":False, "West[1]":True, "GhostScary":True, "West[3]":False, "South[2]":True, "East[1]":False}
    >>> actions = ['North', 'South', 'East', 'West']
    >>> plan = extractActionSequence(model, actions)
    >>> print plan
    ['West', 'South', 'North']
    """

    parse = PropSymbolExpr.parseExpr
    actionAndTime = sorted([parse(key) for key in model if model[key]
                            and parse(key)[0] in actions], key=lambda x: int(x[1]))

    return [a[0] for a in actionAndTime]


def pacmanSuccessorStateAxioms(x, y, t, walls_grid):
    """
    Successor state axiom for state (x,y,t) (from t-1), given the board (as a
    grid representing the wall locations).
    Current <==> (previous position at time t-1) & (took action to move to x, y)
    """
    PosAndAction = [(x + 1, y, 'West'), (x - 1, y, 'East'),
                    (x, y + 1, 'South'), (x, y - 1, 'North')]
    valid = [PropSymbolExpr(pacman_str, x, y, t - 1) & PropSymbolExpr(action, t - 1)
             for x, y, action in PosAndAction if not walls_grid[x][y]]
    return to_cnf(PropSymbolExpr(pacman_str, x, y, t) % disjoin(valid))


def positionLogicPlan(problem, isFood=False, st=None, gl=lambda t: None):
    """
    Given an instance of a PositionPlanningProblem, return a list of actions that lead to the goal.
    Available actions are game.Directions.{NORTH,SOUTH,EAST,WEST}
    Note that STOP is not an available action.
    """
    walls, width, height = problem.walls, problem.getWidth(), problem.getHeight()
    start = st if isFood else problem.getStartState()
    actions = ['South', 'North', 'West', 'East']
    allPos = list(product(range(1, width+1), range(1, height+1)))
    validPos = [pos for pos in allPos if not walls[pos[0]][pos[1]]]

    initial = PropSymbolExpr(pacman_str, *start, 0)
    other = [~PropSymbolExpr(pacman_str, *pos, 0)
             for pos in allPos if pos != start]
    constraints = conjoin(initial, *other)

    t = 0
    model = False
    while not model:
        oneAction = [exactlyOne([PropSymbolExpr(a, t) for a in actions])]
        successor = [pacmanSuccessorStateAxioms(x, y, t + 1, walls)
                     for x, y in validPos]
        constraints = conjoin(constraints, *oneAction, *successor)
        goal = gl(t) if isFood else PropSymbolExpr(
            pacman_str, *problem.getGoalState(), t + 1)
        model = findModel(conjoin(constraints, goal))
        t += 1

    return extractActionSequence(model, actions)


def foodLogicPlan(problem):
    """
    Given an instance of a FoodPlanningProblem, return a list of actions that help Pacman
    eat all of the food.
    Available actions are game.Directions.{NORTH,SOUTH,EAST,WEST}
    Note that STOP is not an available action.
    """
    start, foods = problem.getStartState()

    def foodConstraints(t):
        return conjoin([atLeastOne([PropSymbolExpr(pacman_str, *pos, past)
                                    for past in range(t+1)]) for pos in foods.asList()])

    return positionLogicPlan(problem, True, start, foodConstraints)


# Abbreviations
plp = positionLogicPlan
flp = foodLogicPlan

# Some for the logic module uses pretty deep recursion on long expressions
sys.setrecursionlimit(100000)
