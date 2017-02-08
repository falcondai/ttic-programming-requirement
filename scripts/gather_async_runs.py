#!/usr/bin/env python

import glob, os, argparse

parser = argparse.ArgumentParser()
parser.add_argument('-l', '--log-dir', type=str, required=True)
parser.add_argument('-p', '--port', type=int, default=6006)

args = parser.parse_args()

# we can point tensorboard to the chief event directories in each run individually
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tensorboard/README.md

paths_to_chief = glob.glob(os.path.join(args.log_dir, '*', 'worker-0'))
name_paths = []
tb_logdir_str = ''
print 'found runs:'
for p in paths_to_chief:
    run_name = p.split(os.path.sep)[-2]
    print run_name, p
    name_paths.append((run_name, p))

tb_logdir_str = ','.join(['%s:%s' % x for x in name_paths])

os.system('tensorboard --logdir %s --port %s' % (tb_logdir_str, str(args.port)))
