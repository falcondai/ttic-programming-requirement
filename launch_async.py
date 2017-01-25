import argparse, os, sys


def new_tmux_cmd(name, cmd):
    if isinstance(cmd, (list, tuple)):
        cmd = ' '.join(str(v) for v in cmd)
    return name, "tmux send-keys -t {} '{}' Enter".format(name, cmd)


def create_tmux_commands(session, num_workers, env_id, logdir):
    # for launching the TF workers and for launching tensorboard
    base_cmd = [
        'CUDA_VISIBLE_DEVICES=', sys.executable, 'async_node.py',
        '--log-dir', logdir, '--env-id', env_id,
        '--n-workers', str(num_workers), '--clip-norm', str(100.)]

    # parameter server
    cmds_map = [new_tmux_cmd('ps', base_cmd + ['--job', 'ps'])]
    # workers
    for i in range(num_workers):
        cmds_map += [new_tmux_cmd(
            'w-%d' % i, base_cmd + ['--job', 'worker', '--task-index', str(i)])]

    # tensorboard
    cmds_map += [new_tmux_cmd('tb', ['tensorboard --logdir {} --port 12345'.format(logdir)])]
    # htop
    cmds_map += [new_tmux_cmd('htop', ['htop'])]

    windows = [v[0] for v in cmds_map]

    cmds = [
        'mkdir -p {}'.format(logdir),
        'tmux kill-session',
        'tmux new-session -s {} -n {} -d'.format(session, windows[0]),
    ]
    for w in windows[1:]:
        cmds += ['tmux new-window -t {} -n {}'.format(session, w)]
    cmds += ['sleep 1']
    for window, cmd in cmds_map:
        cmds += [cmd]

    return cmds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run commands')
    parser.add_argument('-w', '--n-workers', default=1, type=int,
                        help="Number of workers")
    parser.add_argument('-e', '--env-id', type=str, default='PongDeterministic-v3',
                        help="Environment id")
    parser.add_argument('-l', '--log-dir', type=str, default="/tmp/pong",
                        help="Log directory path")

    args = parser.parse_args()

    cmds = create_tmux_commands('a3c', args.n_workers, args.env_id, args.log_dir)
    print('\n'.join(cmds))
    os.system('\n'.join(cmds))
