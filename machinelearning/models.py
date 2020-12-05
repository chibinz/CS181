import nn
from itertools import chain
from functools import partial


class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        return nn.DotProduct(x, self.w)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        return 1 if nn.as_scalar(self.run(x)) >= 0 else -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        converged = False
        while not converged:
            converged = True
            for x, y in dataset.iterate_once(1):
                if nn.as_scalar(y) != self.get_prediction(x):
                    converged = False
                    self.w.update(x, nn.as_scalar(y))


class GenericNNModel(object):
    def __init__(self, widths, lossFunction, batchSize, learningRate, targetAccuracy):
        dimension = [(widths[i], widths[i+1]) for i in range(0, len(widths)-1)]
        self.weight = list(map(lambda x: nn.Parameter(*x), dimension))
        self.bias = list(map(lambda x: nn.Parameter(1, x[1]), dimension))
        self.lossFunction = lossFunction
        self.batchSize = batchSize
        self.learningRate = learningRate
        self.targetAccuracy = targetAccuracy

    def run(self, x):
        layer = x
        for w, b in list(zip(self.weight, self.bias))[:-1]:
            layer = nn.ReLU(nn.AddBias(nn.Linear(layer, w), b))

        return nn.AddBias(nn.Linear(layer, self.weight[-1]), self.bias[-1])

    def get_loss(self, x, y):
        return self.lossFunction(self.run(x), y)

    def train(self, dataset):
        try:
            dataset.get_validation_accuracy()
        except Exception:
            def get_accuracy(s):
                sample = list(map(lambda p: nn.as_scalar(
                    self.get_loss(*p)), s.iterate_once(self.batchSize)))
                return 1 - sum(sample) / len(sample)
            dataset.get_validation_accuracy = partial(get_accuracy, dataset)

        while dataset.get_validation_accuracy() < self.targetAccuracy:
            for x, y in dataset.iterate_once(self.batchSize):
                gradients = nn.gradients(
                    self.get_loss(x, y), self.weight + self.bias)
                for param, grad in zip(chain(self.weight, self.bias), gradients):
                    param.update(grad, -self.learningRate)


class RegressionModel(GenericNNModel):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """

    def __init__(self):  # 2 layers
        super().__init__([1, 64, 1], nn.SquareLoss, 100, 0.1, 0.98)


class DigitClassificationModel(GenericNNModel):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self):  # 2 layers
        super().__init__([784, 400, 10], nn.SoftmaxLoss, 100, 0.5, 0.975)


class LanguageIDModel(GenericNNModel):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self):  # Fix to 3 layers
        super().__init__([47, 200, 200, 5], nn.SoftmaxLoss, 100, 0.1, 0.85)

    def run(self, xs):
        layer = nn.Linear(nn.DataNode(xs[0].data), self.weight[0])
        for x in xs:
            layer = nn.ReLU(nn.AddBias(nn.Linear(
                nn.Add(nn.Linear(x, self.weight[0]), layer), self.weight[1]), self.bias[1]))
        return nn.AddBias(nn.Linear(layer, self.weight[2]), self.bias[2])
