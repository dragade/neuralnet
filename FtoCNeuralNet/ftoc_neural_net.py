#!/usr/bin/env python3
"""
FtoCNeuralNet - the smallest possible neural network, explained.

It learns to convert Fahrenheit to Celsius by looking at examples.
We never tell it the formula  C = (F - 32) * 5/9.  It has to discover it.

This file is the command-line program: it reads the options, draws the
ASCII graphs, and prints the lesson step by step. The neural network
itself (the neuron, the math, the training loop) is in neuron.py.
Read that file first if you want to see how the learning works.

Only the Python standard library is used, so every step is visible.
"""

import argparse
import sys
import time

from neuron import (
    INPUT_SCALE,
    TEST_TEMPERATURES_F,
    TRAINING_DATA,
    Neuron,
    train,
    true_celsius,
)


# --------------------------------------------------------------------------
# 1. Drawing and printing helpers
# --------------------------------------------------------------------------
def heading(text):
    print()
    print("=" * 72)
    print(text)
    print("=" * 72)


def ascii_plot(neuron, width=61, height=17):
    """Draw the training points and the neuron's current line.

        o  = a training example (the right answer)
        .  = the neuron's current prediction line
        @  = the line passes through a training example
    """
    # The region of the graph we draw, in degrees.
    f_min, f_max = -50.0, 220.0
    c_min, c_max = -50.0, 110.0

    # Convert a temperature into a character position on the grid.
    # Row 0 is the TOP of the screen, so higher Celsius means a smaller row.
    def col(f):
        return round((f - f_min) / (f_max - f_min) * (width - 1))

    def row(c):
        return round((c_max - c) / (c_max - c_min) * (height - 1))

    grid = [[" "] * width for _ in range(height)]

    # Draw the neuron's line: ask it for a prediction at every column.
    for i in range(width):
        f = f_min + i * (f_max - f_min) / (width - 1)
        r = row(neuron.predict(f))
        if 0 <= r < height:
            grid[r][i] = "."

    # Draw the training examples on top of the line.
    for f, c in TRAINING_DATA:
        r, i = row(c), col(f)
        grid[r][i] = "@" if grid[r][i] == "." else "o"

    lines = []
    for r in range(height):
        c_value = c_max - r * (c_max - c_min) / (height - 1)
        label = f"{c_value:6.0f}C |" if r % 4 == 0 else "        |"
        lines.append(label + "".join(grid[r]))
    lines.append("        +" + "-" * width)
    lines.append(f"        {f_min:.0f}F" + " " * (width - 12) + f"{f_max:.0f}F")
    return "\n".join(lines)


def status_line(epoch, neuron):
    return (f"epoch {epoch:>5}   w = {neuron.w:9.4f}   b = {neuron.b:9.4f}   "
            f"loss = {neuron.loss(TRAINING_DATA):12.4f}")


def snapshot_epochs(epochs):
    """Epochs worth showing: frequent early on, when changes are biggest."""
    marks = {0, 1, 2, 3, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500,
             1000, 2000, 5000, 10000, 20000, 50000, 100000}
    return {e for e in marks if e <= epochs} | {epochs}


def make_epoch_display(epochs, animate, delay):
    """Build the function that draws the graph during training.

    The training loop in neuron.py calls it after every epoch. It either
    redraws every frame (--animate) or prints a snapshot at chosen epochs.
    """
    show_at = snapshot_epochs(epochs)

    def show_epoch(epoch, neuron):
        if animate:
            sys.stdout.write("\033[H\033[J")  # clear the terminal
            print(status_line(epoch, neuron))
            print(ascii_plot(neuron))
            time.sleep(delay)
        elif epoch in show_at:
            print()
            print(status_line(epoch, neuron))
            print(ascii_plot(neuron))

    return show_epoch


# --------------------------------------------------------------------------
# 2. The lesson, step by step
# --------------------------------------------------------------------------
def show_introduction(neuron, learning_rate, epochs):
    heading("FtoCNeuralNet: teaching one neuron to convert F to C")
    print("""
The network is a single neuron with two numbers it can learn:

    x = F / {scale:g}
    prediction = w * x + b

    w (weight) : how strongly the input affects the output
    b (bias)   : a constant shift added to every answer

Both start at 0. We never give it the formula. It only sees these examples:
""".format(scale=neuron.input_scale))
    print("      Fahrenheit   Celsius (target)")
    for f, c in TRAINING_DATA:
        print(f"      {f:10.1f}   {c:10.2f}")
    print(f"""
Settings: learning rate = {learning_rate}, epochs = {epochs}
(An epoch is one pass over all {len(TRAINING_DATA)} examples followed by one update.)""")


def show_first_step_in_detail(neuron, learning_rate):
    """Perform epoch 1 by hand, printing every number along the way."""
    heading("A close look at epoch 1: forward pass, loss, gradients, update")
    print("\nStep 1 - FORWARD PASS: run every example through the neuron.\n")
    print("      F        x=F/scale   prediction   target    error (pred - target)")
    for f, c in TRAINING_DATA:
        x = neuron.scale_input(f)
        p = neuron.predict(f)
        print(f"  {f:7.1f}   {x:9.3f}   {p:10.3f}   {c:7.2f}   {p - c:10.3f}")

    print(f"\nStep 2 - LOSS: average squared error = {neuron.loss(TRAINING_DATA):.3f}")
    print("         (big number = very wrong; the goal is to push it toward 0)")

    grad_w, grad_b = neuron.gradients(TRAINING_DATA)
    print("\nStep 3 - GRADIENTS: which way should w and b move?")
    print(f"         d(loss)/dw = {grad_w:9.3f}")
    print(f"         d(loss)/db = {grad_b:9.3f}")
    print("         Negative gradient -> increasing the value lowers the loss.")

    old_w, old_b = neuron.w, neuron.b
    neuron.train_step(TRAINING_DATA, learning_rate)
    print("\nStep 4 - UPDATE: new value = old value - learning_rate * gradient")
    print(f"         w = {old_w:.4f} - {learning_rate} * ({grad_w:.3f}) = {neuron.w:.4f}")
    print(f"         b = {old_b:.4f} - {learning_rate} * ({grad_b:.3f}) = {neuron.b:.4f}")
    print(f"\nNew loss: {neuron.loss(TRAINING_DATA):.3f}  "
          "(it went down, so the neuron improved)")
    print("\nTraining repeats these 4 steps again and again.")


def show_explosion(epoch, neuron):
    print(f"\n{status_line(epoch, neuron)}")
    print("\nThe loss EXPLODED. The learning rate is too large for these "
          "inputs,\nso each step overshoots further than the last. "
          "Try a smaller --lr\nor leave input normalization on.")


def show_results(neuron):
    heading("What did the neuron learn?")
    slope = neuron.slope_per_degree_f()
    print(f"""
  Learned:   C = {slope:.4f} * F + ({neuron.b:.4f})
  Real:      C = 0.5556 * F + (-17.7778)      [same as (F - 32) * 5/9]

  The weight became the slope (5/9 = 0.5556) and the bias became the
  offset (-32 * 5/9 = -17.78). The neuron discovered the formula
  from examples alone.
""")
    print("Test on temperatures it never saw during training:\n")
    print("      F        network says    real answer    off by")
    for f in TEST_TEMPERATURES_F:
        guess, real = neuron.predict(f), true_celsius(f)
        print(f"  {f:7.1f}      {guess:9.2f} C    {real:9.2f} C    {abs(guess - real):6.3f}")


def interactive(neuron):
    if not sys.stdin.isatty():
        return
    print("\nTry it yourself! Enter a Fahrenheit temperature (blank line to quit).")
    while True:
        try:
            text = input("  F> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            break
        try:
            f = float(text)
        except ValueError:
            print("  Please enter a number.")
            continue
        print(f"  network: {neuron.predict(f):.2f} C   (real: {true_celsius(f):.2f} C)")


# --------------------------------------------------------------------------
# 3. Command-line entry point
# --------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a single neuron to convert Fahrenheit to Celsius.")
    parser.add_argument("--lr", type=float, default=0.1,
                        help="learning rate: how big each update step is (default 0.1)")
    parser.add_argument("--epochs", type=int, default=200,
                        help="number of training passes (default 200)")
    parser.add_argument("--animate", action="store_true",
                        help="redraw the plot every epoch, like a movie")
    parser.add_argument("--delay", type=float, default=0.05,
                        help="seconds between frames with --animate (default 0.05)")
    parser.add_argument("--no-normalize", action="store_true",
                        help="feed raw Fahrenheit values in (try it with the default --lr!)")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    return args


def main():
    args = parse_args()

    # A scale of 1.0 means the raw Fahrenheit value goes straight in.
    neuron = Neuron(input_scale=1.0 if args.no_normalize else INPUT_SCALE)

    # The lesson: introduce, walk through one step slowly, train, check.
    show_introduction(neuron, args.lr, args.epochs)
    print("\nStarting point (w = 0, b = 0, so every prediction is 0):\n")
    print(ascii_plot(neuron))
    show_first_step_in_detail(neuron, args.lr)

    heading("Training: watch the line ('.') move onto the examples ('o')")
    show_epoch = make_epoch_display(args.epochs, args.animate, args.delay)
    show_epoch(1, neuron)  # epoch 1 was done by hand in the walkthrough

    exploded_at = train(neuron, TRAINING_DATA, args.lr, args.epochs,
                        on_epoch=show_epoch, start_epoch=2)
    if exploded_at is not None:
        show_explosion(exploded_at, neuron)
        return

    show_results(neuron)
    interactive(neuron)


if __name__ == "__main__":
    main()
