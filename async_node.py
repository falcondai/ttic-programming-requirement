#!/usr/bin/env python
import tensorflow as tf
import argparse, os, importlib

from envs import get_env
from a3c import A3C, add_arguments

class FastSaver(tf.train.Saver):
    # HACK disable saving metagraphs
    def save(self, sess, save_path, global_step=None, latest_filename=None, meta_graph_suffix='meta', write_meta_graph=True):
        super(FastSaver, self).save(sess, save_path, global_step, latest_filename, meta_graph_suffix, False)


def build_cluster(n_workers, ps_port=2220):
    host = 'localhost'
    cluster = tf.train.ClusterSpec({
        'ps': ['%s:%d' % (host, ps_port)],
        'worker': ['%s:%d' % (host, ps_port + p + 1) for p in xrange(n_workers)],
        })
    return cluster

def run(args, server, env, build_model):
    summary_dir = os.path.join(args.log_dir, 'worker-%i' % args.task_index)
    checkpoint_dir = os.path.join(args.log_dir, 'checkpoints')
    writer = tf.summary.FileWriter(summary_dir, flush_secs=30)

    print '* environment spec:'
    print env.spec
    trainer = A3C(env.spec, env.reset, env.step, build_model, args.task_index, writer, args)

    # save non-local variables
    variables_to_save = [v for v in tf.global_variables() if not v.name.startswith('local')]
    init_op = tf.variables_initializer(variables_to_save)
    saver = FastSaver(variables_to_save,
                      keep_checkpoint_every_n_hours=1,
                      max_to_keep=2)
    # save metagraph
    if args.task_index == 0:
        saver.export_meta_graph(os.path.join(checkpoint_dir, 'model.meta'))

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
    print '* supervisor %i stopped' % args.task_index

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', choices=['ps', 'worker'], default='worker')
    parser.add_argument('--n-workers', default=2, type=int)
    parser.add_argument('--task-index', default=0, type=int)
    parser.add_argument('-e', '--env-id', type=str, default='atari.skip.quarter.Pong')
    parser.add_argument('-m', '--model', type=str, default='cnn_gru_pi_v')
    parser.add_argument('--log-dir', type=str, default='/tmp/pongd')

    # add additional A3C arguments
    add_arguments(parser)

    args = parser.parse_args()

    cluster = build_cluster(args.n_workers)
    config = tf.ConfigProto(gpu_options={
            'allow_growth': True,
        }, intra_op_parallelism_threads=2)
    if args.job == 'worker':
        # worker
        server = tf.train.Server(cluster, job_name='worker', task_index=args.task_index, config=config)
        model = importlib.import_module('models.%s' % args.model)
        env = get_env(args.env_id)
        run(args, server, env, model.build_model)
    else:
        # parameter server
        server = tf.train.Server(cluster, job_name='ps', task_index=args.task_index, config=config)
        server.join()
