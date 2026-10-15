"""
neuron.py - the neural network itself.

This file holds everything that makes the network learn: the training
data, the neuron, and the training loop. It does no printing or drawing;
that lives in ftoc_neural_net.py. You could reuse this file in a different
program, such as a web page or a notebook, without changing it.

The whole "network" is ONE neuron:

        input            weight            bias          output
     F / 100  ───────►  ×  w   ───────►   +  b   ───────►  predicted C

So the neuron computes:   prediction = w * x + b      (where x = F / 100)

Training means: start with w and b at 0, then nudge them over and over
in whichever direction makes the predictions less wrong.

Only the Python standard library is used, so every step is visible.
"""

import math

# --------------------------------------------------------------------------
# 1. Training data: (Fahrenheit, Celsius) pairs the network learns from.
# --------------------------------------------------------------------------
TRAINING_DATA = [
    (-40.0, -40.0),   # where the two scales meet
    (0.0, -17.78),
    (32.0, 0.0),      # water freezes
    (50.0, 10.0),
    (72.0, 22.22),    # room temperature
    (100.0, 37.78),
    (212.0, 100.0),   # water boils
]

# Temperatures the network never sees during training.
# If it gets these right, it learned the rule, not just the examples.
TEST_TEMPERATURES_F = [98.6, 350.0, -10.0, 451.0]

# NORMALIZATION: dividing inputs by 100 keeps the numbers small (roughly
# -1 to 2). Why does that matter? The gradient for the weight is multiplied
# by the input (see Neuron.gradients). A raw input like 212 makes that
# gradient about 200 times larger than the bias gradient, so any learning
# rate that suits the bias makes the weight overshoot wildly. Small inputs
# keep both gradients a similar size, and one learning rate works for both.
# Real neural networks normalize their inputs for the same reason.
# Run with --no-normalize to see what goes wrong without it.
INPUT_SCALE = 100.0

# If the loss grows past this, training has blown up and won't recover.
EXPLODED_LOSS = 1e12


def true_celsius(f):
    """The real formula, used only to check the network's answers."""
    return (f - 32.0) * 5.0 / 9.0


# --------------------------------------------------------------------------
# 2. The neuron
# --------------------------------------------------------------------------
class Neuron:
    """A single artificial neuron with one input and a linear output.

    Real networks stack thousands of these and pass each output through an
    "activation function" (such as ReLU or sigmoid) so they can learn curves.
    Converting F to C is a straight-line relationship, so this neuron needs
    no activation function. Its output is just w * x + b.
    """

    def __init__(self, input_scale=INPUT_SCALE):
        # The PARAMETERS: the only things the network can change while it
        # learns. Everything the neuron "knows" lives in these two numbers.
        # They start at 0, so the neuron knows nothing and every
        # prediction is 0 at first. (Big networks usually start with small
        # random numbers instead, but zero works fine for a single neuron.)
        self.w = 0.0
        self.b = 0.0
        self.input_scale = input_scale

    def scale_input(self, fahrenheit):
        """Normalize the raw temperature into the small number x the neuron sees."""
        return fahrenheit / self.input_scale

    def predict(self, fahrenheit):
        """Forward pass: input -> multiply by weight -> add bias -> output."""
        x = self.scale_input(fahrenheit)
        return self.w * x + self.b

    def loss(self, data):
        """Mean squared error: average of (prediction - target)^2.

        Squaring makes every error positive and punishes big misses more.
        A loss of 0 would mean every prediction is perfect.
        """
        total = 0.0
        for f, c in data:
            error = self.predict(f) - c
            total += error ** 2
        return total / len(data)

    def gradients(self, data):
        """How much would the loss change if we nudged w or b?

        Picture the loss as a valley. The gradient tells us which way is
        uphill, and how steep it is, from where we stand right now.

        For one example:   error = w*x + b - target
                           loss  = error^2
        The chain rule from calculus gives:
            d(loss)/dw = 2 * error * (d error / dw) = 2 * error * x
            d(loss)/db = 2 * error * (d error / db) = 2 * error * 1
        We average these over all examples.

        In words: the weight's blame is scaled by its input. If x was big,
        w had a big effect on the mistake, so w gets a bigger correction.

        A positive gradient means "increasing this makes things worse",
        so we will move the opposite way. Big networks compute the same
        kind of gradients for millions of weights; that process is called
        BACKPROPAGATION.
        """
        grad_w = 0.0
        grad_b = 0.0
        for f, c in data:
            x = self.scale_input(f)
            error = self.predict(f) - c
            grad_w += 2 * error * x
            grad_b += 2 * error
        n = len(data)
        return grad_w / n, grad_b / n

    def train_step(self, data, learning_rate):
        """One step of GRADIENT DESCENT: take a small step downhill on the loss.

        The LEARNING RATE sets how big that step is:
          - too small: learning is very slow
          - too large: each step jumps past the bottom of the valley and
            lands higher on the other side, so the loss explodes
        Try --lr 0.01 and --lr 0.8 to see both cases.
        """
        grad_w, grad_b = self.gradients(data)
        # Subtract, because the gradient points uphill and we want downhill.
        self.w -= learning_rate * grad_w
        self.b -= learning_rate * grad_b
        return grad_w, grad_b

    def slope_per_degree_f(self):
        """The weight converted back to 'Celsius per Fahrenheit degree'.

        The neuron saw F / 100, so its weight is 100 times the real slope.
        Dividing by the scale undoes that, so we can compare it with 5/9.
        """
        return self.w / self.input_scale


# --------------------------------------------------------------------------
# 3. The training loop
# --------------------------------------------------------------------------
def train(neuron, data, learning_rate, epochs, on_epoch=None, start_epoch=1):
    """Run gradient descent from epoch `start_epoch` through epoch `epochs`.

    This short loop is the heart of all neural network training:
    predict, measure the error, compute gradients, update, repeat.

    on_epoch: an optional function called as on_epoch(epoch, neuron) after
              every update. The command-line program uses it to draw graphs,
              which keeps all the printing out of this file.

    Returns None if training finished normally, or the epoch number where
    the loss exploded (a sign the learning rate is too large).
    """
    for epoch in range(start_epoch, epochs + 1):
        neuron.train_step(data, learning_rate)

        # If the learning rate is too big, the loss grows every step until
        # it overflows to infinity or NaN ("not a number").
        loss = neuron.loss(data)
        if math.isnan(loss) or loss > EXPLODED_LOSS:
            return epoch

        if on_epoch is not None:
            on_epoch(epoch, neuron)
    return None
