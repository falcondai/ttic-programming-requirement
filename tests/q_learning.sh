#!/bin/bash
python train_q.py --model simple2_q --checkpoint_dir checkpoints/test-q-rms --optimizer rmsprop --n_train_steps 2000 --env CartPole-v0 --momentum 0 --initial_learning_rate 1e-2 --n_update_episodes 4 --n_batch_ticks 128 --n_lr_decay_steps 1000 --lr_decay_rate 0.8 --initial_epsilon 0.2 --epsilon_decay_rate 0.01
