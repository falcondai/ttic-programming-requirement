#!/bin/bash
python train_policy_gradient.py --model simple --checkpoint_dir checkpoints/test-pg-rms --optimizer rmsprop --n_train_steps 1000 --env CartPole-v0 --momentum 0 --initial_learning_rate 1e-2 --n_update_episodes 4 --n_batch_ticks 128 --n_decay_steps 1000 --decay_rate 0.8
