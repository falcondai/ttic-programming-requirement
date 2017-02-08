# TODO

## implementation
- refactor TD(lambda) and eligibility traces
- change arg parser to partial parse
- implement continuous action models
- add tuple type for state (gym has StateTuple)
- human agent (needs to be env-specific, doom -> spectator mode, atari -> WASD, etc)
- algorithmic envs

## envs
- mujoco
- mountain-car
- discrete ob space games (maze)
- mnist glimpse
- mnist/cifar-10 compression
- rubik's cube
- adversarial games

## experiments
- gym classic env's with FF and RNN
- gym atari env's with RNN
- test on multi-GPU machines
- FF slower than RNN?
- try motion blur with RNN
