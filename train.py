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
    get_current_run_id, restore_vars


def process_rollout(rewards, reward_gamma, next_state_value=0.):
    acc_reward = next_state_value
    partial_rewards_to_go = []
    for reward in rewards[::-1]:
        acc_reward = reward + reward_gamma * acc_reward
        partial_rewards_to_go.insert(0, acc_reward)
    return partial_rewards_to_go

def train(train_env, args, build_model):
    env_spec, env_step, env_reset, env_render = train_env

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
            obs_ph, state_prev, keep_prob_ph, rnn_state, action_probs, \
            state_value, rnn = build_model(policy_input_shape,
                                           env_spec['action_size'])

        actions_taken_ph = tf.placeholder('int32')
        target_value_ph = tf.placeholder('float')

        # entropy regularizer to encourage action diversity
        log_action_probs = tf.log(action_probs)
        action_entropy = - tf.reduce_mean(tf.reduce_sum(action_probs \
                                                        * log_action_probs, 1))
        action_logits = vector_slice(log_action_probs, actions_taken_ph)

        # objective for value estimation
        value_objective = tf.reduce_mean(tf.square(target_value_ph \
                                                   - state_value))

        # objective for computing policy gradient
        state_advantage = target_value_ph - state_value
        policy_objective = tf.reduce_mean(action_logits * \
                                          tf.stop_gradient(state_advantage))

        # total objective
        # maximize policy objective and minimize value objective
        # and maximize action entropy
        objective = - policy_objective + args['value_objective_coeff'] \
            * value_objective - args['action_entropy_coeff'] * action_entropy

        # optimization
        global_step = tf.Variable(0, trainable=False, name='global_step')
        # global tick keeps track of ticks experienced by the agent
        global_tick = tf.Variable(0, trainable=False, name='global_tick')
        delta_tick = tf.placeholder('int32')
        increment_global_tick = global_tick.assign_add(1)

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
        grad_vars = optimizer.compute_gradients(objective)
        update_op = optimizer.apply_gradients(grad_vars,
                                              global_step=global_step)

        # summaries
        episode_len_ph = tf.placeholder('float')
        episode_reward_ph = tf.placeholder('float')
        ticks_per_second_ph = tf.placeholder('float')
        steps_per_second_ph = tf.placeholder('float')
        images_ph = tf.placeholder('uint8')

        per_episode_summary = tf.summary.merge([
            tf.summary.scalar('episodic/reward', episode_reward_ph),
            tf.summary.scalar('episodic/length', episode_len_ph),
            tf.summary.scalar('episodic/reward_per_tick',
                              episode_reward_ph / episode_len_ph),
            tf.summary.scalar('episodic/ticks_per_second',
                              ticks_per_second_ph)
        ])

        grad_summaries = []
        grads = []
        var_list = []
        print '* extra summary'
        for g, v in grad_vars:
            grad_summaries.append(tf.summary.histogram('gradients/%s' % v.name, g))
            print 'gradients/%s' % v.name
            grads.append(g)
            var_list.append(v)

        # normed_grads, norm = tf.clip_by_global_norm(grads, 40.)
        # update_op = optimizer.apply_gradients(list(zip(normed_grads, var_list)),
        #                                       global_step=global_step)
        # normed_gn = tf.global_norm(normed_grads)
        norm = tf.global_norm(grads)

        per_step_summary = tf.summary.merge(grad_summaries + [
            tf.summary.scalar('model/learning_rate', learning_rate),
            tf.summary.scalar('model/objective', objective),
            tf.summary.scalar('model/state_value_objective',
                              value_objective),
            tf.summary.scalar('model/policy_objective', policy_objective),
            tf.summary.scalar('model/action_entropy', action_entropy),
            tf.summary.scalar('model/gradient_norm', norm),
            # tf.summary.scalar('model/gradient_norm_after_clip', normed_gn),
            tf.summary.scalar('model/steps_per_second', steps_per_second_ph),
            tf.image_summary('frames', images_ph, max_images=3),
        ])

        saver = tf.train.Saver(max_to_keep=2, \
                               keep_checkpoint_every_n_hours=1)
        with tf.Session() as sess:
            writer = tf.summary.FileWriter(summary_dir, sess.graph,
                                           flush_secs=30)
            print '* writing summary to', summary_dir

            tf.train.export_meta_graph('%s/model.meta' % args['checkpoint_dir'])
            restore_vars(saver, sess, args['checkpoint_dir'], args['restart'])

            # print '* regularized parameters:'
            # for v in tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES):
            #     print v.name

            # stochastic policy
            policy_func = lambda obs, h: action_probs.eval(feed_dict={
                obs_ph: [obs],
                state_prev: h,
                keep_prob_ph: 1. - args['dropout_rate'],
            })[0]

            state_value_func = lambda obs, h: state_value.eval(feed_dict={
                obs_ph: [obs],
                state_prev: h,
                keep_prob_ph: 1. - args['dropout_rate'],
            })[0,0]

            h = rnn.zero_state(1, 'float')
            n_ticks = args['n_update_ticks']
            n_obs_ticks = args['n_obs_ticks']
            obs = []
            actions = []
            rewards = []
            rewards_to_go = []
            terminals = []
            done = True
            episode_start = None
            tick = global_tick.eval()
            delta_tick, step = 0, 0
            step_start = time.time()
            current_episode_start = 0
            progress_bar = tqdm.trange(args['n_train_steps']).__iter__()
            while True:
                # on-policy rollout
                if done:
                    # calculate rewards-to-go till the end of last episode
                    rewards_to_go += process_rollout(rewards[current_episode_start:], args['reward_gamma'])

                    # reset the observations queue
                    observations = [np.zeros(env_spec['observation_shape'])]\
                        * (n_obs_ticks - 1)
                    observations.append(env_reset())
                    h = rnn.zero_state(1, 'float')
                    current_episode_start = delta_tick

                    # episode time stats
                    if episode_start != None:
                        episode_dt = time.time() - episode_start
                        # per-episode summary
                        per_episode_summary_val = per_episode_summary.eval({
                            episode_reward_ph: episode_reward,
                            episode_len_ph: episode_len,
                            ticks_per_second_ph: episode_len / episode_dt,
                        })
                        writer.add_summary(per_episode_summary_val, tick)
                    # reset episode stats
                    episode_start = time.time()
                    episode_len = 0
                    episode_reward = 0.

                model_input = np.concatenate(observations[-n_obs_ticks:], \
                                             axis=-1)
                obs.append(model_input)
                # sample action according to policy
                action = np.random.choice(env_spec['action_size'], \
                                          p=policy_func(model_input, h))
                h = rnn_state.eval(feed_dict={
                    obs_ph: model_input,
                    state_prev: h,
                })
                actions.append(action)

                next_obs, reward, done = env_step(action)
                if env_render != None:
                    env_render()
                observations.append(next_obs)
                rewards.append(reward)
                if episode_len >= env_spec['timestep_limit'] - 1:
                    # enforce time limit
                    done = True
                terminals.append(done)

                episode_reward += reward
                episode_len += 1
                delta_tick += 1
                tick = increment_global_tick.eval()

                if delta_tick == args['n_update_ticks']:
                    # got enough ticks for an update
                    if terminals[-1]:
                        rewards_to_go += process_rollout(rewards[current_episode_start:], args['reward_gamma'])
                    else:
                        # use the next state's value to
                        # estimate rewards-to-go
                        next_model_input = np.concatenate(observations[-n_obs_ticks:], \
                                                     axis=-1)
                        next_state_value = state_value_func(next_model_input, h)
                        rewards_to_go += process_rollout(rewards[current_episode_start:], args['reward_gamma'], next_state_value)

                    step_dt = time.time() - step_start
                    step_start = time.time()
                    per_step_summary_val, _ = sess.run([per_step_summary, update_op], feed_dict={
                        obs_ph: obs,
                        target_value_ph: rewards_to_go,
                        actions_taken_ph: actions,
                        steps_per_second_ph: 1 / step_dt,
                        images_ph: observations[-n_obs_ticks:],
                        })
                    # write per-step summary
                    writer.add_summary(per_step_summary_val, tick)

                    observations = observations[-n_obs_ticks:]
                    obs = []
                    actions = []
                    rewards = []
                    terminals = []
                    rewards_to_go = []
                    step += 1
                    delta_tick = 0
                    current_episode_start = 0
                    progress_bar.next()
                    if step % args['n_save_interval'] == 0:
                        saver.save(sess, args['checkpoint_dir'] + '/model',
                                   global_step=global_step.eval(), write_meta_graph=False)

                    if step >= args['n_train_steps']:
                        break

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

    parse.add_argument('--action_entropy_coeff', type=float, default=0.01)
    parse.add_argument('--value_objective_coeff', type=float, default=0.1)
    parse.add_argument('--reward_gamma', type=float, default=1.)
    parse.add_argument('--dropout_rate', type=float, default=0.2)

    parse.add_argument('--restart', action='store_true')
    parse.add_argument('--checkpoint_dir', required=True)
    parse.add_argument('--summary_prefix', default='')
    parse.add_argument('--render', action='store_true')

    # how many episodes to rollout before update parameters
    parse.add_argument('--n_update_ticks', type=int, default=256)
    parse.add_argument('--n_batch_ticks', type=int, default=128)
    parse.add_argument('--n_save_interval', type=int, default=8)
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
    parse.add_argument('--initial_learning_rate', type=float, default=0.01)
    parse.add_argument('--n_decay_steps', type=int, default=512)
    parse.add_argument('--no_decay_staircase', action='store_true')
    parse.add_argument('--decay_rate', type=float, default=0.8)

    parse.add_argument('--np_seed', type=int, default=123)
    parse.add_argument('--tf_seed', type=int, default=1234)

    return parse


if __name__ == '__main__':
    from functools import partial
    from util import passthrough, use_render_state, scale_env, atari_env

    # arguments
    parse = build_argparser()
    args = parse.parse_args()

    gym_env = gym.make(args.env)

    if args.use_render_state:
        env_spec, env_step, env_reset, env_render = use_render_state(
            gym_env, args.scale, args.interpolation)
    else:
        env_spec, env_step, env_reset, env_render = passthrough(gym_env)
        if len(env_spec['observation_shape']) == 3 and args.scale != 1.:
            # the observation space is an image
            # env_spec, env_step, env_reset, env_render = scale_env((env_spec, env_step, env_reset, env_render), args.scale, args.interpolation)
            env_spec, env_step, env_reset, env_render = atari_env((env_spec, env_step, env_reset, env_render), args.scale, 1)


    env_spec['timestep_limit'] = min(gym_env.spec.timestep_limit,
                                     args.timestep_limit)
    env_render = env_render if args.render else None

    print '* environment', args.env
    print 'observation shape', env_spec['observation_shape']
    print 'action space', env_spec['action_size']
    print 'timestep limit', env_spec['timestep_limit']
    print 'reward threshold', gym_env.spec.reward_threshold

    # model
    model = importlib.import_module('models.%s' % args.model)

    # train
    # gym monitor
    if args.monitor:
        env.monitor.start(args['monitor_dir'])

    train((env_spec, env_step, env_reset, env_render),
          vars(args),
          model.build_model)

    if args.monitor:
        env.monitor.close()
