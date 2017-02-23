#!/bin/bash
python launch_tmux_async.py -l /tmp/ac-planar-0 -a adv_ac.ff_fc.1 -e planar.2 --learning-rate 1e-3 -w 4 --action-entropy-coeff 3.
