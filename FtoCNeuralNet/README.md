# FtoCNeuralNet

The smallest possible neural network: one neuron that learns to convert
Fahrenheit to Celsius from examples, without being told the formula.

It is written for students. It uses only the Python standard library, so
every step of the math appears in the code and in the output.

## Files

| File | What's in it |
|---|---|
| `neuron.py` | The neural network: training data, the `Neuron` class (predict, loss, gradients, update) and the training loop. No printing. **Start reading here.** |
| `ftoc_neural_net.py` | The command-line program: options (argparse), ASCII graphs and the step-by-step lesson output. |
| `web_ui.py` | The browser version: a tiny local web server that trains with `neuron.py` and sends the results to the page. |
| `web_ui.html` | The browser page: charts that play back the training. It only draws; all learning happens in Python. |
| `explained.html` | A beginner's guide (written for 8th graders) to how the neuron learns, with hands-on knobs. Open it directly in a browser; no server needed. |

## Run it

```bash
python3 ftoc_neural_net.py
```

Requires Python 3.6+. There is nothing to install.

The program will:

1. Show the training examples and the neuron's equation (`prediction = w * x + b`).
2. Walk through one training step in detail: forward pass, loss, gradients and update.
3. Train, drawing an ASCII graph so you can watch the line move onto the examples.
4. Show the learned formula next to the real one and test it on new temperatures.
5. Let you type temperatures and see the network's answers.

## Run it in the browser

```bash
python3 web_ui.py
```

This opens http://127.0.0.1:8000/ in your browser (use `--port` to pick another
port and `--no-browser` to skip opening it). Press Ctrl+C in the terminal to stop.

The page shows:

- **The neuron**: the forward pass with live numbers. Type any temperature to try it.
- **How wrong is it?**: loss (MSE), RMSE in °C, the two gradients and the learned formula.
- **The line moving onto the examples**: the prediction line, with each example's error drawn as a bar.
- **The loss landscape**: a map of the loss for every (w, b) pair, with the path gradient descent walks downhill.
- **Loss over time**: on a log scale. Click it to jump to an epoch.
- **Temperatures it never saw**: a check that it learned the rule.

Use Play/Pause, the arrow buttons (or ← → keys) and the slider to scrub through
training. The learning-rate presets show slow, good and exploding training.
Uncheck **Normalize input** to see raw inputs make the default learning rate
explode, then try learning rate 0.0001 with 50000 epochs.

## Experiments to try

| Command | What you'll learn |
|---|---|
| `python3 ftoc_neural_net.py --animate` | Watch the line move each epoch, like a movie |
| `python3 ftoc_neural_net.py --lr 0.01` | A small learning rate is slow; the line has not finished moving after 200 epochs |
| `python3 ftoc_neural_net.py --lr 0.01 --epochs 2000` | ...but it gets there with enough time |
| `python3 ftoc_neural_net.py --lr 0.8` | A large learning rate overshoots and the loss explodes |
| `python3 ftoc_neural_net.py --epochs 5` | Stop early: the network is only partly trained |
| `python3 ftoc_neural_net.py --no-normalize` | Without input scaling, raw values like 212 make training blow up |
| `python3 ftoc_neural_net.py --no-normalize --lr 0.0001 --epochs 50000` | It can work without scaling, but needs a tiny step and about 50,000 epochs |

## Key ideas

- **Weight (w)** and **bias (b)**: the only numbers the network learns.
- **Forward pass**: compute a prediction from the input.
- **Loss**: one number that says how wrong the predictions are (mean squared error).
- **Gradient**: which direction to nudge each parameter to lower the loss.
- **Gradient descent**: repeatedly take small steps downhill on the loss.
- **Learning rate**: how big each step is.
- **Epoch**: one pass over all the training examples.
- **Normalization**: keeping inputs small so training is stable.

## Next steps

This problem is a straight line, so one neuron is enough. Problems like XOR
need curved or bent boundaries. Those need hidden layers and activation
functions, which are the next ideas to learn.
