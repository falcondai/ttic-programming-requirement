# async training launch script meant for use on slurm-managed clusters
# additionally, we can launch a bunch of them at once by
# - piping the script output to files
# - submit all of the scripts in a folder with find
#   `$ find *.sh -exec sbatch -p contrib-cpu {} \;`

import argparse, os, sys

def slurm_cmd(cmd):
    return ' '.join(str(v) for v in cmd)

def create_slurm_commands(num_workers, logdir, use_gpu, port, extra_args):
    # for launching the TF workers and for launching tensorboard
    base_cmd = [
        sys.executable, 'async_node.py', '--log-dir', logdir, '--n-workers', num_workers, '--cluster-port', port + 1]

    if not use_gpu:
        # hide GPU from tensorflow
        base_cmd = ['CUDA_VISIBLE_DEVICES='] + base_cmd

    cmds = []
    # parameter server
    cmd = slurm_cmd(base_cmd + ['--job', 'ps'])
    cmds.append(cmd)

    # workers
    for i in range(num_workers):
        cmd = slurm_cmd(base_cmd + ['--job', 'worker', '--task-index', i] + extra)
        cmds.append(cmd)

    return cmds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-w', '--n-workers', default=1, type=int, help='number of workers')
    parser.add_argument('-l', '--log-dir', type=str, default='/tmp/pong', help='checkpoint directory path')
    parser.add_argument('-g', '--use-gpu', action='store_true', help='use GPU for training')
    parser.add_argument('-p', '--port', type=int, help='port for tensorboard', default=12345)

    args, extra = parser.parse_known_args()

    cmds = create_slurm_commands(args.n_workers, args.log_dir, args.use_gpu, args.port, extra)

    # generate and print the bash script for `sbatch`
    # can be piped into `sbatch` command to execute directly
    print '#!/bin/sh'
    # helpful job name
    print '#SBATCH -J %s' % os.path.basename(args.log_dir)
    # request 2 * n_workers CPU cores
    print '#SBATCH -c %i' % (2 * args.n_workers)
    for cmd in cmds:
        # cancel the whole job if one of the processes fails
        print '(', cmd, ';', 'scancel $SLURM_JOB_ID', ')', '&'
        # magic
        print 'sleep 1s'
    print 'wait'
