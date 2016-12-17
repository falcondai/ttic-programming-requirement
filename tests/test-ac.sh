#!/bin/bash
dai@photon:~/dev/ml/policy-gradient$ python train_actor_critic.py --checkpoint_dir checkpoints/ac-test --model shared_pi_v --restart --momentum 0 --initial_learning_rate 0.01 --n_update_ticks 1024
