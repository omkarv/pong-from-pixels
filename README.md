## Introduction

This repo trains a Reinforcement Learning Neural Network so that it's able to play Pong from raw pixel input.

I've written up a [blog post](https://medium.com/@omkarv/intro-to-reinforcement-learning-pong-92a94aa0f84d) which walks through the code here and the basic principles of Reinforcement Learning, with Pong as the guiding
example.

It is largely based on [a Gist by Andrej Karpathy](https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5), which
in turn is based on the [Playing Atari with Deep Reinforcement Learning paper by Mnih et al.](https://arxiv.org/abs/1312.5602)

This script uses the [Open AI Gym environments](https://github.com/openai/gym) in order to run the Atari emulator and environments, and currently uses no external ML framework & only numpy.

## The AI Agent Pong in action

### Prior to training (mostly random actions)
![Prior to training (mostly random actions)](https://github.com/omkarv/pong-from-pixels/blob/master/experiment-output/base-init.gif)

### After training base repo + learning rate modification
![After training](https://github.com/omkarv/pong-from-pixels/blob/master/experiment-output/base-after-overnight-train.gif)

The agent that played this game was trained for ~12000 episodes (basically 12000 episodes of 'best-of-21' rounds) over a period of ~ 15 hours, on a Macbook Pro 2018 with 2.6GHz i7 (6 cores).  The running mean score per episode, over the trailing 100 episodes, at the point I stopped training was -5, i.e. the CPU would win each episode 21-16 on average.

**Hyperparameters:**
* Default except for learning-rate 1e-3

### After training base repo + learning rate modification + a bugfix

A minor fix was added which crops more of the image vs the base repo, by removing noisy parts of the image where we can safely ignore the ball motion. This boosted the observed performance and speed at which the AI beat the CPU on average (i.e. when the average reward for an episode exceeded 0)

**Hyperparameters:**
* Default except for learning-rate 1e-3

The agent that played this game was trained for ~10000 episodes (basically 10000 episodes of 'best-of-21' rounds) over a period of ~ 13 hours, on a Macbook Pro 2018 with 2.6GHz i7 (6 cores).  The running mean score per episode, over the trailing 100 episodes, at the point I stopped training was 2.5, i.e. the trained AI Agent would win each episode 21 points to 18.5.

Training for another 10 hours & another 5000 episodes allowed the trained AI Agent to reach a running mean score per epsisode of 5, i.e. the trained AI Agent would win each episode 21 points to 16.

**Graph of reward over time - first 10000 episodes of training**
![Reward over time with bugfix](https://github.com/omkarv/pong-from-pixels/blob/master/experiment-output/bugfix-rewards-chart.png)

**Graph of reward over time - 10000 to 15000 episodes of training**

![Reward over time after 10000 episodes](https://github.com/omkarv/pong-from-pixels/blob/master/experiment-output/bugfix-rewards-chart-after-10000.png)
### Modifications vs Source Gist
* Records output video of the play
* Modified learning rate from 1e-4 to 1e-3
* Comments for clarity
* Minor fix which crops more of the image vs the base repo

### Installation Requirements
The instructions below are for Mac OS & assume you have Homebrew installed.

* You'll need to run the code with Python 2.7 - I recommend the use of `conda` to manage python environments
* Install Open AI Gym `brew install gym`
* Install Cmake `brew install cmake`
* Install ffmpeg `brew install ffmpeg` - Required for monitoring / videos

