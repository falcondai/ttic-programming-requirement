#!/usr/bin/env python

import argparse, os, sys


def new_tmux_cmd(name, cmd):
    if isinstance(cmd, (list, tuple)):
        cmd = ' '.join(str(v) for v in cmd)
    return name, "tmux send-keys -t {} '{}' Enter".format(name, cmd)

def create_tmux_commands(session, num_workers, logdir, use_gpu, port, extra_args):
    # for launching the TF workers and for launching tensorboard
    # cluster start from the next port
    base_cmd = [sys.executable, 'async_node.py', '--log-dir', logdir, '--n-workers', num_workers, '--cluster-port', port + 1]

    if not use_gpu:
        # hide GPU from tensorflow
        base_cmd = ['CUDA_VISIBLE_DEVICES='] + base_cmd

    # parameter server
    cmds_map = [new_tmux_cmd('ps', base_cmd + ['--job', 'ps'])]
    # workers
    for i in range(num_workers):
        cmds_map += [new_tmux_cmd( 'w-%d' % i, base_cmd + ['--job', 'worker', '--task-index', str(i)] + extra)]

    # tensorboard
    cmds_map += [new_tmux_cmd('tb', ['tensorboard --logdir {} --port {}'.format(logdir, port)])]

    windows = [v[0] for v in cmds_map]

    cmds = [
        'mkdir -p {}'.format(logdir),
        'tmux new-session -s {} -n {} -d'.format(session, windows[0]),
    ]
    for w in windows[1:]:
        cmds += ['tmux new-window -t {} -n {}'.format(session, w)]
    cmds += ['sleep 1']
    for window, cmd in cmds_map:
        cmds += [cmd]

    return cmds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-w', '--n-workers', default=1, type=int, help='number of workers')
    parser.add_argument('-l', '--log-dir', type=str, default='/tmp/pong', help='checkpoint directory path')
    parser.add_argument('-g', '--use-gpu', action='store_true', help='use GPU for training')
    parser.add_argument('-p', '--port', type=int, help='port for tensorboard', default=12345)

    args, extra = parser.parse_known_args()

    cmds = create_tmux_commands(os.path.basename(args.log_dir), args.n_workers, args.log_dir, args.use_gpu, args.port, extra)
    print('\n'.join(cmds))
    os.system('\n'.join(cmds))
