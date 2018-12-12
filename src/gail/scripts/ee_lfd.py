import argparse
import tempfile
import os.path as osp
import gym
from tqdm import tqdm
from mpi4py import MPI

import tensorflow as tf
import util as U_
import numpy as np

import mlp_policy
from ur_dset import UR_Dset
from adversary import Adversary

from robot_env import RobotEnv

from baselines.common import set_global_seeds, tf_util as U
from baselines.common.misc_util import boolean_flag
from baselines.common.mpi_adam import MpiAdam

def argsparser():
    parser = argparse.ArgumentParser("Tensorflow Implementation of End-effector Learning from Demonstration")
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--task', type=str, choices=['train', 'evaluate'], default='train')
    parser.add_argument('--expert_path', type=str, default=U_.getDataPath() + '/obs_acs.npz')
    parser.add_argument('--checkpoint_dir', help='the directory to save model', default=U_.getPath() + '/checkpoint')
    parser.add_argument('--log_dir', help='the directory to save log file', default=U_.getPath() + '/log')
    # Network Configuration (Using MLP Policy)
    parser.add_argument('--policy_hidden_size', type=int, default=100)
    parser.add_argument('--adversary_hidden_size', type=int, default=100)
    # Optimization Configuration
    parser.add_argument('--g_step', help='number of steps to train policy in each epoch', type=int, default=3)
    parser.add_argument('--d_step', help='number of steps to train discriminator in each epoch', type=int, default=1)
    # Algorithms Configuration
    parser.add_argument('--max_kl', type=float, default=0.01)
    parser.add_argument('--policy_entcoeff', help='entropy coefficiency of policy', type=float, default=0)
    parser.add_argument('--adversary_entcoeff', help='entropy coefficiency of discriminator', type=float, default=1e-3)
    # Traing Configuration
    parser.add_argument('--save_per_iter', help='save model every xx iterations', type=int, default=100)
    parser.add_argument('--num_timesteps', help='number of timesteps per episode', type=int, default=5e6)
    # for evaluatation
    boolean_flag(parser, 'evaluate', default=False, help='use stochastic/deterministic policy to evaluate')
    boolean_flag(parser, 'stochastic_policy', default=False, help='use stochastic/deterministic policy to evaluate')
    return parser.parse_args()

def bc_learn(bool_evaluate, robot, policy_func, dataset, optim_batch_size=64, max_iters=5*1e3,
          adam_epsilon=1e-5, optim_stepsize=3e-4,
          ckpt_dir=None, log_dir=None, task_name=None,
          verbose=False):

    val_per_iter = int(max_iters/10)
    pi = policy_func("pi", robot.observation_space, robot.action_space)  # Construct network for new policy
    saver = tf.train.Saver()

    if bool_evaluate:
        saver.restore(tf.get_default_session(), U_.getPath() + '/model/bc.ckpt')
        return pi

    # placeholder
    ob = U.get_placeholder_cached(name="ob")
    ac = pi.pdtype.sample_placeholder([None])
    stochastic = U.get_placeholder_cached(name="stochastic")
    loss = tf.reduce_mean(tf.square(ac-pi.ac))
    var_list = pi.get_trainable_variables()
    adam = MpiAdam(var_list, epsilon=adam_epsilon)
    lossandgrad = U.function([ob, ac, stochastic], [loss]+[U.flatgrad(loss, var_list)])

    U.initialize()
    adam.sync()
    print("Pretraining with Behavior Cloning...")
    for iter_so_far in tqdm(range(int(max_iters))):
        ob_expert, ac_expert = dataset.get_next_batch(optim_batch_size, 'train')
        train_loss, g = lossandgrad(ob_expert, ac_expert, True)
        adam.update(g, optim_stepsize)
        if verbose and iter_so_far % val_per_iter == 0:
            ob_expert, ac_expert = dataset.get_next_batch(-1, 'val')
            val_loss, _ = lossandgrad(ob_expert, ac_expert, True)
            print("Training loss: {}, Validation loss: {}".format(train_loss, val_loss))
            saver.save(tf.get_default_session(), 'model/bc.ckpt')

    return pi

def trpo_train(env, seed, policy_fn, reward_giver, dataset,
               g_step, d_step, policy_entcoeff, num_timesteps, save_per_iter,
               checkpoint_dir, log_dir, task_name=None):

    import trpo
    # Set up for MPI seed
    rank = MPI.COMM_WORLD.Get_rank()
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)
    U.initialize()
    trpo.learn(env, policy_fn, reward_giver, dataset, rank,
               g_step=g_step, d_step=d_step,
               entcoeff=policy_entcoeff,
               ckpt_dir=checkpoint_dir, log_dir=log_dir,
               save_per_iter=save_per_iter,
               timesteps_per_batch=1024,
               max_timesteps=num_timesteps,
               max_kl=0.01, cg_iters=10, cg_damping=0.1,
               gamma=0.995, lam=0.97,
               vf_iters=5, vf_stepsize=1e-3,
               task_name=task_name)

def runner(env, policy_func, load_model_path, timesteps_per_batch, number_trajs,
           stochastic_policy, save=False, reuse=False):

    # Setup network
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_func("pi", ob_space, ac_space, reuse=reuse)
    U.initialize()
    # Prepare for rollouts
    # ----------------------------------------
    U.load_state(load_model_path)

    obs_list = []
    acs_list = []
    len_list = []
    ret_list = []
    for _ in tqdm(range(number_trajs)):
        traj = traj_1_generator(pi, env, timesteps_per_batch, stochastic=stochastic_policy)
        obs, acs, ep_len, ep_ret = traj['ob'], traj['ac'], traj['ep_len'], traj['ep_ret']
        obs_list.append(obs)
        acs_list.append(acs)
        len_list.append(ep_len)
        ret_list.append(ep_ret)
    if stochastic_policy:
        print('stochastic policy:')
    else:
        print('deterministic policy:')
    if save:
        filename = load_model_path.split('/')[-1] + '.' + env.spec.id
        np.savez(filename, obs=np.array(obs_list), acs=np.array(acs_list),
                 lens=np.array(len_list), rets=np.array(ret_list))
    avg_len = sum(len_list)/len(len_list)
    avg_ret = sum(ret_list)/len(ret_list)
    print("Average length:", avg_len)
    print("Average return:", avg_ret)
    return avg_len, avg_ret

def main(args):
    robot = RobotEnv()
    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(args.seed)

    def policy_fn(name, ob_space, ac_space, reuse=False):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                    reuse=reuse, hid_size=args.policy_hidden_size, num_hid_layers=3)

    task_name = U_.get_task_name(args)
    args.checkpoint_dir = osp.join(args.checkpoint_dir, task_name)
    args.log_dir = osp.join(args.log_dir, task_name)
    dataset = UR_Dset(expert_path=args.expert_path)

    ob_shape = dataset.get_ob_shape()
    ac_shape = dataset.get_ac_shape()

    robot.set_ob_shape(ob_shape)
    robot.set_ac_shape(ac_shape)

    '''
    pi = bc_learn(args.evaluate,
               robot,
               policy_fn,
               dataset,
               ckpt_dir=args.checkpoint_dir,
               log_dir=args.log_dir,
               task_name=task_name,
               verbose=True)

    from time import sleep
    ob = robot.reset()
    sleep(2)

    # testing directly mapping
    for _ in tqdm(range(1000)):
        _, acs = dataset.get_next_batch(1, 'train')
        _ = robot.step(acs[0])
        # sleep(0.5)
    
    # testing policy
    for _ in tqdm(range(1000)):
        ac, _ = pi.act(True, ob)
        ob = robot.step(ac)
        # sleep(0.5)
    '''

    if args.task == 'train':
        reward_giver = Adversary(robot, args.adversary_hidden_size)
        trpo_train(robot,
                   args.seed,
                   policy_fn,
                   reward_giver,
                   dataset,
                   args.g_step,
                   args.d_step,
                   args.policy_entcoeff,
                   args.num_timesteps,
                   args.save_per_iter,
                   args.checkpoint_dir,
                   args.log_dir,
                   task_name
                   )
    elif args.task == 'evaluate':
        runner(robot,
               policy_fn,
               args.load_model_path,
               timesteps_per_batch=1024,
               number_trajs=10,
               stochastic_policy=args.stochastic_policy,
               save=args.save_sample
               )
    else:
        raise NotImplementedError

if __name__ == '__main__':
    args = argsparser()
    main(args)
