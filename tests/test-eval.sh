#!/bin/bash
python eval.py --checkpoint_path checkpoints/test-q-rms --latest --policy greedy --model q
python eval.py --checkpoint_path checkpoints/test-sarsa-rms --latest --policy greedy --model q
python eval.py --checkpoint_path checkpoints/test-pg-rms --latest --policy greedy --model pi
