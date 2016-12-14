#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import os, sys, cPickle, time, itertools, json
from Queue import deque
import tqdm
import argparse
import importlib
import gym
from util import pad_zeros, duplicate_obs, vector_slice, \
    get_current_run_id, restore_vars, rollout

def sample_ticks(behavior_policy, env_spec, env_step, env_reset, last_obs_q,
            last_done=True, env_render=None, n_update_ticks=256, n_obs_ticks=1):
    '''rollout based on behavior policy from an environment'''

    observations, actions, rewards, nonterminals = [], [], [], []
    t = 0
    done = last_done
    obs_q = last_obs_q
    obs = last_obs_q[-1]
    while t < n_update_ticks:
        if done or t >= env_spec['timestep_limit']:
            obs = env_reset()
            # pad the first observation with zeros
            obs_q = deque(pad_zeros([obs], n_obs_ticks), n_obs_ticks)
        policy_input = np.concatenate(obs_q, axis=-1)
        action_probs = behavior_policy(policy_input)
        action = np.random.choice(env_spec['action_size'], p=action_probs)
        obs_q.popleft()
        observations.append(obs)
        actions.append(action)
        obs, reward, done = env_step(action)
        nonterminals.append([0. if done else 1.])
        rewards.append([reward])
        obs_q.append(obs)
        if env_render != None:
            env_render()
        t += 1
    return observations, actions, rewards, nonterminals, done, obs_q

def train(train_env, eval_env, args, build_model):
    env_spec, env_step, env_reset, env_render = train_env
    eval_env_spec, eval_env_step, eval_env_reset, eval_env_render = eval_env

    summary_dir = 'tf-log/%s%d-%s' % (args['summary_prefix'], time.time(),
                                      os.path.basename(args['checkpoint_dir']))

    # set seeds
    np.random.seed(args['np_seed'])
    tf.set_random_seed(args['tf_seed'])

    # create checkpoint dirs
    if not os.path.exists(args['checkpoint_dir']):
        os.makedirs(args['checkpoint_dir'])

    print '* training hyperparameters:'
    for k in sorted(args.keys()):
        print k, args[k]
    n_run = get_current_run_id(args['checkpoint_dir'])
    with open('%s/hyperparameters.%i.json' % (args['checkpoint_dir'], n_run),
              'wb') as hpf:
        json.dump(args, hpf)

    with tf.Graph().as_default() as g:
        # model
        print '* building model %s' % args['model']
        policy_input_shape = list(env_spec['observation_shape'])
        policy_input_shape[-1] *= args['n_obs_ticks']
        with tf.variable_scope('model'):
            obs_ph, keep_prob_ph, action_probs, state_value = build_model(
                policy_input_shape,
                env_spec['action_size'])
        with tf.variable_scope('model', reuse=True):
            next_obs_ph, _, _, next_state_value = build_model(
                policy_input_shape,
                env_spec['action_size'])
        actions_taken_ph = tf.placeholder('int32')
        reward_ph = tf.placeholder('float')
        nonterminal_ph = tf.placeholder('float')

        avg_v_objective_ph = tf.placeholder('float')
        avg_len_episode_ph = tf.placeholder('float')
        avg_episode_reward_ph = tf.placeholder('float')
        max_episode_reward_ph = tf.placeholder('float')
        min_episode_reward_ph = tf.placeholder('float')
        avg_tick_reward_ph = tf.placeholder('float')
        avg_action_entropy_ph = tf.placeholder('float')

        # expected reward under policy
        # entropy regularizer to encourage action diversity
        action_entropy = - tf.reduce_mean(tf.reduce_sum(action_probs \
                                                     * tf.log(action_probs), 1))
        action_logits = vector_slice(tf.log(action_probs), actions_taken_ph)

        # objective for value estimation
        target = reward_ph + nonterminal_ph * args['reward_gamma'] \
            * next_state_value
        value_objective = 0.01 * tf.reduce_sum(tf.square(tf.stop_gradient(target) \
                                                  - state_value))

        # objective for computing policy gradient
        state_advantage = target - state_value
        policy_objective = tf.reduce_sum(action_logits * \
                                         tf.stop_gradient(state_advantage)) \
            + args['reg_coeff'] * action_entropy


        # optimization
        global_step = tf.Variable(0, trainable=False, name='global_step')
        learning_rate = tf.train.exponential_decay(
            args['initial_learning_rate'],
            global_step, args['n_decay_steps'],
            args['decay_rate'],
            staircase=not args['no_decay_staircase'])

        if args['optimizer'] == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate,
                                               args['adam_beta1'],
                                               args['adam_beta2'],
                                               args['adam_epsilon'])
        elif args['optimizer'] == 'ag':
            optimizer = tf.train.MomentumOptimizer(learning_rate,
                                                   args['momentum'],
                                                   use_nesterov=True)
        elif args['optimizer'] == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(learning_rate,
                                                   args['rmsprop_decay'],
                                                   args['momentum'],
                                                   args['rmsprop_epsilon'])
        else:
            optimizer = tf.train.MomentumOptimizer(learning_rate,
                                                   args['momentum'])

        # train ops
        # maximize the PG objective
        pg_grad_vars = optimizer.compute_gradients(-policy_objective)
        pg_grad_vars = [(g, v) for g, v in pg_grad_vars if g != None]
        update_policy_op = optimizer.apply_gradients(pg_grad_vars,
                                                     global_step=global_step)

        # minimize the value objective
        # no increment to `global_step`
        v_grad_vars = optimizer.compute_gradients(value_objective)
        v_grad_vars = [(g, v) for g, v in v_grad_vars if g != None]
        update_v_op = optimizer.apply_gradients(v_grad_vars)

        # summaries
        eval_summary_op = tf.merge_summary([
            tf.scalar_summary('average_episode_reward', avg_episode_reward_ph),
            tf.scalar_summary('max_episode_reward', max_episode_reward_ph),
            tf.scalar_summary('min_episode_reward', min_episode_reward_ph),
            tf.scalar_summary('average_episode_length', avg_len_episode_ph),
            ])

        train_summaries = []
        print '* extra summary'
        for g, v in pg_grad_vars:
            train_summaries.append(tf.histogram_summary('pg_gradients/%s' % v.name, g))
            print 'pg_gradients/%s' % v.name
        for g, v in v_grad_vars:
            train_summaries.append(tf.histogram_summary('v_gradients/%s' % v.name, g))
            print 'v_gradients/%s' % v.name

        train_summary_op = tf.merge_summary(train_summaries + [
            tf.scalar_summary('learning_rate', learning_rate),
            tf.scalar_summary('average_action_entropy', avg_action_entropy_ph),
            tf.scalar_summary('average_tick_reward', avg_tick_reward_ph),
            tf.scalar_summary('average_v_objective', avg_v_objective_ph),
            ])

        saver = tf.train.Saver(max_to_keep=2, keep_checkpoint_every_n_hours=1)
        with tf.Session() as sess:
            writer = tf.train.SummaryWriter(summary_dir, sess.graph,
                                            flush_secs=30)
            print '* writing summary to', summary_dir
            restore_vars(saver, sess, args['checkpoint_dir'], args['restart'])

            print '* regularized parameters:'
            for v in tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES):
                print v.name

            # stochastic policy
            policy = lambda obs: action_probs.eval(feed_dict={
                obs_ph: [obs],
                keep_prob_ph: 1. - args['dropout_rate'],
            })[0]

            last_done = True
            last_obs_q = deque(pad_zeros([np.zeros(env_spec['observation_shape'])],
                                   args['n_obs_ticks']), args['n_obs_ticks'])
            for i in tqdm.tqdm(xrange(args['n_train_steps'])):
                # on-policy rollout for some ticks
                observations, actions, rewards, nonterminals, \
                last_done, _last_obs_q = sample_ticks(
                    policy,
                    env_spec,
                    env_step,
                    env_reset,
                    deque(last_obs_q),
                    last_done,
                    env_render,
                    n_update_ticks=args['n_update_ticks'] + 1,
                    n_obs_ticks=args['n_obs_ticks'],
                )
                avg_tick_reward = np.mean(rewards)

                # transform and preprocess the rollouts
                obs = list(duplicate_obs(list(last_obs_q)[:-1] + observations,
                                         args['n_obs_ticks']))
                next_obs = obs[1:]
                # ignore the last tick
                obs = obs[:-1]

                # estimate policy gradient by batches
                # accumulate gradients over batches
                pg_acc_grads = dict([(grad, np.zeros(grad.get_shape()))
                                     for grad, var in pg_grad_vars])
                v_acc_grads = dict([(grad, np.zeros(grad.get_shape()))
                                    for grad, var in v_grad_vars])
                acc_reg = 0.
                acc_v_obj_val = 0.
                n_batch = int(np.ceil(args['n_update_ticks'] * 1. / args['n_batch_ticks']))
                for j in xrange(n_batch):
                    start = j * args['n_batch_ticks']
                    end = min(start + args['n_batch_ticks'], args['n_update_ticks'])
                    grad_feed = {
                        obs_ph: obs[start:end],
                        keep_prob_ph: 1. - args['dropout_rate'],
                        actions_taken_ph: actions[start:end],
                        reward_ph: rewards[start:end],
                        nonterminal_ph: nonterminals[start:end],
                        next_obs_ph: next_obs[start:end],
                    }

                    # compute the expectation of gradients
                    v_obj_val, pg_grad_vars_val, v_grad_vars_val, \
                    action_entropy_val = sess.run([
                        value_objective,
                        pg_grad_vars,
                        v_grad_vars,
                        action_entropy,
                        ], feed_dict=grad_feed)
                    for (g, _), (g_val, _) in zip(pg_grad_vars, pg_grad_vars_val):
                        pg_acc_grads[g] += g_val / args['n_update_ticks']
                    for (g, _), (g_val, _) in zip(v_grad_vars, v_grad_vars_val):
                        v_acc_grads[g] += g_val / args['n_update_ticks']
                    # TODO change regularizer
                    acc_reg += action_entropy_val * (end - start)
                    acc_v_obj_val += v_obj_val
                last_obs_q = _last_obs_q

                # evaluation
                if i % args['n_eval_interval'] == 0:
                    episode_rewards = []
                    episode_lens = []
                    for j in xrange(args['n_eval_episodes']):
                        _, _, er = rollout(
                            policy,
                            eval_env_spec,
                            eval_env_step,
                            eval_env_reset,
                            None,
                            args['n_obs_ticks'],
                        )
                        episode_rewards.append(np.sum(er))
                        episode_lens.append(len(er))
                    eval_summary_val = eval_summary_op.eval({
                        avg_episode_reward_ph: np.mean(episode_rewards),
                        max_episode_reward_ph: np.max(episode_rewards),
                        min_episode_reward_ph: np.min(episode_rewards),
                        avg_len_episode_ph: np.mean(episode_lens),
                    })
                    writer.add_summary(eval_summary_val, global_step.eval())

                # update policy with the sample expectation of gradients
                update_dict = {
                    avg_tick_reward_ph: avg_tick_reward,
                    avg_v_objective_ph: acc_v_obj_val / args['n_update_ticks'],
                    avg_action_entropy_ph: acc_reg / args['n_update_ticks'],
                }
                update_dict.update(pg_acc_grads)
                update_dict.update(v_acc_grads)
                train_summary_val, _, _ = sess.run([train_summary_op,
                                                    update_policy_op,
                                                    update_v_op],
                                                   feed_dict=update_dict)

                writer.add_summary(train_summary_val, global_step.eval())

                if i % args['n_save_interval'] == 0:
                    saver.save(sess, args['checkpoint_dir'] + '/model',
                               global_step=global_step.eval())

            # save again at the end
            saver.save(sess, args['checkpoint_dir'] + '/model',
                       global_step=global_step.eval())

def build_argparser():
    parse = argparse.ArgumentParser()

    parse.add_argument('--model', required=True)

    # gym options
    parse.add_argument('--env', default='CartPole-v0')
    parse.add_argument('--monitor', action='store_true')
    parse.add_argument('--monitor_dir',
                       default='/tmp/gym-monitor-%i' % time.time())
    parse.add_argument('--n_obs_ticks', type=int, default=1)
    parse.add_argument('--timestep_limit', type=int, default=10**9)
    parse.add_argument('--use_render_state', action='store_true')
    parse.add_argument('--scale', type=float, default=1.)
    parse.add_argument('--interpolation', choices=['nearest', 'bilinear',
                                                   'bicubic', 'cubic'],
                       default='bilinear')

    parse.add_argument('--reg_coeff', type=float, default=0.)
    parse.add_argument('--reward_gamma', type=float, default=1.)
    parse.add_argument('--dropout_rate', type=float, default=0.2)

    parse.add_argument('--restart', action='store_true')
    parse.add_argument('--checkpoint_dir', required=True)
    parse.add_argument('--summary_prefix', default='')
    parse.add_argument('--render', action='store_true')

    # how many episodes to rollout before update parameters
    parse.add_argument('--n_update_ticks', type=int, default=256)
    parse.add_argument('--n_batch_ticks', type=int, default=128)
    parse.add_argument('--n_save_interval', type=int, default=1)
    parse.add_argument('--n_train_steps', type=int, default=10**5)
    parse.add_argument('--n_eval_episodes', type=int, default=4)
    parse.add_argument('--n_eval_interval', type=int, default=8)

    # optimizer options
    parse.add_argument('--momentum', type=float, default=0.2)
    parse.add_argument('--adam_beta1', type=float, default=0.9)
    parse.add_argument('--adam_beta2', type=float, default=0.999)
    parse.add_argument('--adam_epsilon', type=float, default=1e-8)
    parse.add_argument('--rmsprop_decay', type=float, default=0.9)
    parse.add_argument('--rmsprop_epsilon', type=float, default=1e-10)

    # training options
    parse.add_argument('--optimizer', choices=['adam', 'momentum', 'ag',
                                               'rmsprop'], default='rmsprop')
    parse.add_argument('--initial_learning_rate', type=float, default=0.001)
    parse.add_argument('--n_decay_steps', type=int, default=512)
    parse.add_argument('--no_decay_staircase', action='store_true')
    parse.add_argument('--decay_rate', type=float, default=0.8)

    parse.add_argument('--np_seed', type=int, default=123)
    parse.add_argument('--tf_seed', type=int, default=1234)

    return parse


if __name__ == '__main__':
    from functools import partial
    from util import passthrough, use_render_state, scale_image

    # arguments
    parse = build_argparser()
    args = parse.parse_args()

    gym_env = gym.make(args.env)
    eval_env = gym.make(args.env)

    if args.use_render_state:
        env_spec, env_step, env_reset, env_render = use_render_state(
            gym_env, args.scale, args.interpolation)
        eval_env_spec, eval_env_step, eval_env_reset, eval_env_render = \
        use_render_state(eval_env, args.scale, args.interpolation)
    else:
        env_spec, env_step, env_reset, env_render = passthrough(gym_env)
        eval_env_spec, eval_env_step, eval_env_reset, eval_env_render = \
        passthrough(eval_env)

    env_spec['timestep_limit'] = min(gym_env.spec.timestep_limit,
                                     args.timestep_limit)
    eval_env_spec['timestep_limit'] = min(eval_env.spec.timestep_limit,
                                     args.timestep_limit)
    env_render = env_render if args.render else None
    eval_env_render = eval_env_render if args.render else None

    print '* environment', args.env
    print 'observation shape', env_spec['observation_shape']
    print 'action space', gym_env.action_space
    print 'timestep limit', env_spec['timestep_limit']
    print 'reward threshold', gym_env.spec.reward_threshold

    # model
    model = importlib.import_module('models.%s' % args.model)

    # train
    # gym monitor
    if args.monitor:
        env.monitor.start(args['monitor_dir'])

    train((env_spec, env_step, env_reset, env_render),
          (eval_env_spec, eval_env_step, eval_env_reset, eval_env_render),
          vars(args),
          model.build_model)

    if args.monitor:
        env.monitor.close()
