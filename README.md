# Reinforcement learning
a tensorflow implementation of asynchronous advantage actor-critic (A3C) method [[1]](https://arxiv.org/abs/1602.01783) and Q-learning.

## dependencies
- python 2.7.x
- OpenCV 2
- tensorflow ^1.0.0 (GPU version optional)
- numpy ^1.12.0
- gym ^0.8.0
- tmux
- ViZDoom (optional) ^1.1.0

For example, on Ubuntu
```bash
# for main features
apt-get install cmake libopencv-dev python-opencv tmux
# for vizdoom (optional)
apt-get install build-essential zlib1g-dev libsdl2-dev libjpeg-dev nasm tar libbz2-dev libgtk2.0-dev git libfluidsynth-dev libgme-dev libopenal-dev timidity libwildmidi-dev libboost-all-dev
```
```bash
pip install tensorflow
# for GPU version tensorflow, use
# pip install tensorflow-gpu
pip install gym vizdoom
pip install gym[atari]
```

## usage
Without writing any code, you may use this project to train an RL agent (via A3C or Q-learning) for solving many environments in OpenAI gym and some standard pre-configured tasks in ViZDoom.

- launch (or resume) asynchronous training (via tmux session): `launch_tmux_async.py`
- test a trained agent: `test.py`
- for details on CLI arguments, use `--help`. For algorithm-specific CLI arguments, check parser definition in trainer source.

To illustrate with a few concrete environments:
- classic pole-balancing task (balancing a pole on a cart by jerking left or right, input is the dynamic state)
  - to visualize the task (with a random agent): `python -m envs.core CartPole-v0 1`
  - to train an A3C agent: `python launch_tmux_async.py -e gym.CartPole-v0 -a adv_ac.ff_fc -l /tmp/cartpole-0 --learning-rate 1e-2`.
    - this launches a tmux session to host a parameter server, a worker and a tensorboard.
    - you may navigate to [localhost:12345](localhost:12345) in a browser to monitor training progress via tensorboard. The main metric to pay attention to is episodic/reward.
    - it takes about a 1m30s to reach optimal behavior. you may kill the tmux session to stop training.
  - to test a trained agent: `python test.py -e gym.CartPole-v0 -a adv_ac.ff_fc -l /tmp/cartpole-0/checkpoints`
  - similarly, to train a Q-learning agent: `python launch_tmux_async.py -e gym.CartPole-v0 -a q.ff_fc -l /tmp/cartpole-q-0 --learning-rate 1e-2 --behavior-policy epsilon`. This takes about 2 min to reach optimal play.
- Atari Pong (a classic console game). The agent observes the pixels on screen and controls a joystick.
  - to visualize the task: `python -m envs.atari skip.Pong`
  - to train an A3C agent: `python launch_tmux_async.py -e atari.skip.quarter.Pong -a adv_ac.cnn_lstm -l /tmp/pong-0 -w 4 --advantage-eval lambda --advantage-lambda 1`
    - this launches four workers processes to speed up rollouts sampling. it takes about 2 hours to converge to optimal play.
  - to test a trained agent: `python test.py -e atari.skip.quarter.Pong -a adv_ac.cnn_lstm -l /tmp/pong-0/checkpoints`
  - many other Atari games are available, such as Breakout and SpaceInvaders. You can replace `Pong` in the above instructions with another game's name.
- ViZDoom (a classic first person shooter Doom). The agent observes the pixels on screen and act in a 3D world. There are several pre-configured tasks available.
  - to visualize the task: `python -m envs.doom simpler_basic`
  - to train an A3C agent: `python launch_tmux_async.py -e doom.simpler_basic -a adv_ac.cnn_gru -l /tmp/doom-sb-0 -w 4 --advantage-eval lambda --advantage-lambda 1`
  - to test a trained agent: `python test.py -e doom.simpler_basic -a adv_ac.cnn_gru -l /tmp/doom-sb-0/checkpoints`

## overview of the implemented algorithms

Please consult algorithm S2 for Q-learning, S3 for A3C in the A3C paper: https://arxiv.org/abs/1602.01783v2

## overview of main implemented abstractions

- `Env`, environment, the abstraction of a Markov decision process (MDP), closely models after `gym.Env`.
  - properties
    - `spec`, the specification of the observation space and action space. Currently, only finite, discrete action space is supported.
  - methods
    - `reset() -> next_observation`, start a new episode.
    - `step(action) -> next_observation, reward, done`, take action in the environment and returns the next observation, instant reward, and whether the episode has ended.
    - `render()`, visualize the environment.
- `Agent`, the actor in the environment. In the case of recurrent models, `StatefulAgent` encapsulates the history state internally.
  - methods
    - `act(observation) -> action`, perform an action based on the input observation.
- `Trainer`, the object responsible for applying a particular training algorithm to update a model with rollouts
  - methods
    - (static) `parser() -> argument_parser`. Returns an argument parser for parsing algorithm-specific arguments from command line.
    - `train(tensorflow_session)`. Use the current agent to obtain a (partial) rollout from environment and update the agent.

## project organization

- `/` various launch scripts
  - `agents/` agents, trainers and registry
  - `models/` neural network model templates (recurrent, feedforward, etc)
  - `envs/` environments and registry
  - `visualize/` visualization utilities

## author
Falcon Dai (dai@ttic.edu)
