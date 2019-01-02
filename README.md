### Introduction

This repo trains a Deep Reinforcement Learning Neural Network, so that's its able to play Pong from raw pixel input.

It is largely based on [this Gist](https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5), which
in turn is based on the [Playing Atari with Deep Reinforcement Learning paper by Mnih et al.](https://arxiv.org/abs/1312.5602)

This script uses the [Open AI Gym environments](https://github.com/openai/gym) in order to run the Atari emulator and environments, and currently uses no external ML framework & only numpy.

### Agent playing Pong in action!

#### Prior to training (mostly random actions) with only UP & DOWN actions
![Prior to training (mostly random actions)](https://github.com/omkarv/pong-from-pixels/blob/master/experiment-output/base-init.gif)

#### After training with only UP & DOWN actions
![After training](https://github.com/omkarv/pong-from-pixels/blob/master/experiment-output/base-after-overnight-train.gif)

The agent that played this game was trained for ~12000 episodes (basically 12000 games up to 21) over a period of ~ 9 hours, on a Macbook Pro 2018 with 2.6GHz i7 (6 cores).  The running mean score per episode, over the trailing 100 episodes, at the point I stopped training was -5, i.e. the CPU would win each episode 21-16 on average.

### Modifications vs Source Gist
* Records output video of the play
* Boosted learning rate from 1e-4 to 1e-3
* Comments for clarity - which is pretty messy atm

### Requirements
The instructions below are for Mac OS & assume you have Homebrew installed.

* You'll need to run the code with Python 2.7
* Install Open AI Gym `brew install gym`
* Install Cmake `brew install cmake`
* Instal ffmpeg `brew install ffmpeg` - Required for monitoring / videos

