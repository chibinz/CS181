# CS181
Artificial Intelligence, UC Berkeley's CS188 taught at ShanghaiTech
![Final Project](uno.png)

## Final
Code, data, and documentation of our final project, an intelligent UNO agent pioneered by my diligent teammates, are made available [here](https://github.com/huiwy/AlphaUNO-Zero). Just to give you a brief idea of what it does, the following is an excerpt from the abstract of our report.
> The game UNO, whose name originated from the Latin "One", has simple rules that could be quickly mastered by a three-year-old. But make no mistake, deeply concealed in the simplicity of rules is the combinatoric complexity of game states. The easy to learn but hard to master characteristic of this game makes it an exciting target for artificial intelligence research. In this paper, we create novel adaptations to Monte Carlo Tree Search, a popular game search technique mostly applied in board games, and tailor it to UNO's highly stochastic nature and complex state space. To reflect on what we have learned in class, we also apply strategies such as informed search, expectimax search, and reinforcement learning in our greedy agent, expectimax agent, and deep Q-learning (DQN) agent. We evaluate the effectiveness of our agents by letting them compete against random agents and among themselves. Our experimental result suggests that our Monte-Carlo Tree Search - Dynamic Bayesian Model (MCTS-DBM) agent surpasses the random agent significantly and beats state-of-the-art UNO agent by a wide margin.

## Tips
Prior to taking this course, I had little experience writing any python programs of significant size (single file scripts of few hundred lines at most). I see the 6 projects of CS188 as both a means of understanding algorithms taught in class and an opportunity to exercise the interesting language features of python.
- Most data presented to you in the 6 projects are in the form of python `list`s. Using `for` loops to iterate over data is an okay solution, but it is by no means concise, elegant, or pythonic. Make good use of high order functions like `map`, `reduce`, `sum`, as well as python `list` comprehensions.
> Simple is better than complex. Flat is better than nested.
- In [project 4](tracking/inference.py), you'll probably need to implement a cache to speed up the computation of particle filtering. Declaring a dictionary is one way, but better way is to make use of python's builtin support. Check out `lru_cache` in `functools`.
```python
@lru_cache
def cachedDistribution(pos):
    return self.getPositionDistribution(gameState, pos)
```
- A common pattern seen among questions is that latter agents improve upon prior agents. That is to say, you don't need to copy around code, just inherit from agents previously implemented. A good example to demonstrate the power of inheritance would be [project 5](machinelearning/models.py), where independent questions largely implement the same functionality with only minor difference in neural network architecture. Command the power of OOP!
```python
class GenericNNModel(object):
    def __init__(self, widths, lossFunction, batchSize, learningRate, targetAccuracy):
        ...

    def run(self, x):
        ...

    def get_loss(self, x, y):
        ...
        return self.lossFunction(self.run(x), y)

    def train(self, dataset):
        ...
```
