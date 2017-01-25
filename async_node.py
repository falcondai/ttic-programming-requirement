#!/usr/bin/env python
import tensorflow as tf
import argparse, os, importlib

from envs.core import GymEnv
from envs.wrappers import GrayscaleWrapper, ScaleWrapper
from a3c import A3C, add_arguments
# from models.cnn_gru_pi_v import build_model

class FastSaver(tf.train.Saver):
    # HACK disable saving metagraphs
    def save(self, sess, save_path, global_step=None, latest_filename=None, meta_graph_suffix='meta', write_meta_graph=True):
        super(FastSaver, self).save(sess, save_path, global_step, latest_filename, meta_graph_suffix, False)


def build_cluster(n_workers):
    host = 'localhost'
    ps_port = 2220
    cluster = tf.train.ClusterSpec({
        'ps': ['%s:%d' % (host, ps_port)],
        'worker': ['%s:%d' % (host, ps_port + p + 1) for p in xrange(n_workers)],
        })
    return cluster

def run(args, server, build_model):
    summary_dir = os.path.join(args.log_dir, 'worker-%i' % args.task_index)
    checkpoint_dir = os.path.join(args.log_dir, 'checkpoints')
    writer = tf.summary.FileWriter(summary_dir, flush_secs=30)

    gym_env = GymEnv(args.env_id)
    env = GrayscaleWrapper(ScaleWrapper(gym_env, args.scale))
    trainer = A3C(env.spec, env.reset, env.step, build_model, args.task_index, writer, args)

    # save non-local variables
    variables_to_save = [v for v in tf.global_variables() if not v.name.startswith('local')]
    init_op = tf.variables_initializer(variables_to_save)
    saver = FastSaver(variables_to_save)

    sv = tf.train.Supervisor(
        is_chief=(args.task_index == 0),
        saver=saver,
        init_op=init_op, # only initialize variables on chief
        ready_op=tf.report_uninitialized_variables(variables_to_save),
        summary_writer=writer, # summarize `global_step / sec`
        summary_op=None,
        global_step=trainer.global_tick,
        logdir=checkpoint_dir,
        save_model_secs=30,
        save_summaries_secs=30,
        )

    config = tf.ConfigProto(gpu_options={
            'allow_growth': True,
        })
    with sv.managed_session(server.target, config=config) as sess, sess.as_default():
        print '* Starting training at global tick', trainer.global_tick.eval()
        while not sv.should_stop():
            trainer.train(sess)

    sv.stop()
    print '* supervisor stopped'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', choices=['ps', 'worker'], default='worker')
    parser.add_argument('--n-workers', default=2, type=int)
    parser.add_argument('--task-index', default=0, type=int)
    parser.add_argument('--env-id', default='PongDeterministic-v3')
    parser.add_argument('--scale', type=float, default=0.5)
    parser.add_argument('--model', type=str, default='cnn_gru_pi_v')
    # parser.add_argument('--interpolation', choices=['nearest', 'bilinear', 'bicubic', 'cubic'], default='bilinear')
    parser.add_argument('--log-dir', type=str, default='/tmp/pongd')

    # add additional A3C arguments
    add_arguments(parser)

    args = parser.parse_args()

    cluster = build_cluster(args.n_workers)
    config = tf.ConfigProto(gpu_options={
            'allow_growth': True,
        })
    if args.job == 'worker':
        # worker
        server = tf.train.Server(cluster, job_name=args.job, task_index=args.task_index, config=config)
        model = importlib.import_module('models.%s' % args.model)
        run(args, server, model.build_model)
    else:
        # parameter server
        server = tf.train.Server(cluster, job_name='ps', task_index=args.task_index, config=config)
        server.join()
