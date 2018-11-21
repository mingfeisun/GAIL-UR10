#!/bin/bash

#PBS -l walltime=72:00:00,nodes=1:ppn=12,mem=10gb
#PBS -N im_modern
#PBS -q atlas
#PBS -o /dev/null
#PBS -e /dev/null

sleep $[ ( $RANDOM % 120 ) + 1 ]s

read -r -d '' COMMANDS << END
python scripts/imitate_mj.py --mode bclone --env Hopper-v1 --data imitation_runs/modern_stochastic/trajs/trajs_hopper.h5 --limit_trajs 4 --data_subsamp_freq 20 --max_iter 20001 --log imitation_runs/modern_stochastic/checkpoints_all/alg=bclone,task=hopper,num_trajs=4,run=0.h5
python scripts/imitate_mj.py --mode bclone --env Hopper-v1 --data imitation_runs/modern_stochastic/trajs/trajs_hopper.h5 --limit_trajs 11 --data_subsamp_freq 20 --max_iter 20001 --log imitation_runs/modern_stochastic/checkpoints_all/alg=bclone,task=hopper,num_trajs=11,run=0.h5
python scripts/imitate_mj.py --mode bclone --env Hopper-v1 --data imitation_runs/modern_stochastic/trajs/trajs_hopper.h5 --limit_trajs 18 --data_subsamp_freq 20 --max_iter 20001 --log imitation_runs/modern_stochastic/checkpoints_all/alg=bclone,task=hopper,num_trajs=18,run=0.h5
python scripts/imitate_mj.py --mode bclone --env Hopper-v1 --data imitation_runs/modern_stochastic/trajs/trajs_hopper.h5 --limit_trajs 25 --data_subsamp_freq 20 --max_iter 20001 --log imitation_runs/modern_stochastic/checkpoints_all/alg=bclone,task=hopper,num_trajs=25,run=0.h5
python scripts/imitate_mj.py --mode bclone --env Walker2d-v1 --data imitation_runs/modern_stochastic/trajs/trajs_walker.h5 --limit_trajs 4 --data_subsamp_freq 20 --max_iter 20001 --log imitation_runs/modern_stochastic/checkpoints_all/alg=bclone,task=walker,num_trajs=4,run=0.h5
python scripts/imitate_mj.py --mode bclone --env Walker2d-v1 --data imitation_runs/modern_stochastic/trajs/trajs_walker.h5 --limit_trajs 11 --data_subsamp_freq 20 --max_iter 20001 --log imitation_runs/modern_stochastic/checkpoints_all/alg=bclone,task=walker,num_trajs=11,run=0.h5
python scripts/imitate_mj.py --mode bclone --env Walker2d-v1 --data imitation_runs/modern_stochastic/trajs/trajs_walker.h5 --limit_trajs 18 --data_subsamp_freq 20 --max_iter 20001 --log imitation_runs/modern_stochastic/checkpoints_all/alg=bclone,task=walker,num_trajs=18,run=0.h5
python scripts/imitate_mj.py --mode bclone --env Walker2d-v1 --data imitation_runs/modern_stochastic/trajs/trajs_walker.h5 --limit_trajs 25 --data_subsamp_freq 20 --max_iter 20001 --log imitation_runs/modern_stochastic/checkpoints_all/alg=bclone,task=walker,num_trajs=25,run=0.h5
python scripts/imitate_mj.py --mode bclone --env Ant-v1 --data imitation_runs/modern_stochastic/trajs/trajs_ant.h5 --limit_trajs 4 --data_subsamp_freq 20 --max_iter 20001 --log imitation_runs/modern_stochastic/checkpoints_all/alg=bclone,task=ant,num_trajs=4,run=0.h5
python scripts/imitate_mj.py --mode bclone --env Ant-v1 --data imitation_runs/modern_stochastic/trajs/trajs_ant.h5 --limit_trajs 11 --data_subsamp_freq 20 --max_iter 20001 --log imitation_runs/modern_stochastic/checkpoints_all/alg=bclone,task=ant,num_trajs=11,run=0.h5
python scripts/imitate_mj.py --mode bclone --env Ant-v1 --data imitation_runs/modern_stochastic/trajs/trajs_ant.h5 --limit_trajs 18 --data_subsamp_freq 20 --max_iter 20001 --log imitation_runs/modern_stochastic/checkpoints_all/alg=bclone,task=ant,num_trajs=18,run=0.h5
python scripts/imitate_mj.py --mode bclone --env Ant-v1 --data imitation_runs/modern_stochastic/trajs/trajs_ant.h5 --limit_trajs 25 --data_subsamp_freq 20 --max_iter 20001 --log imitation_runs/modern_stochastic/checkpoints_all/alg=bclone,task=ant,num_trajs=25,run=0.h5
python scripts/imitate_mj.py --mode bclone --env HalfCheetah-v1 --data imitation_runs/modern_stochastic/trajs/trajs_halfcheetah.h5 --limit_trajs 4 --data_subsamp_freq 20 --max_iter 20001 --log imitation_runs/modern_stochastic/checkpoints_all/alg=bclone,task=halfcheetah,num_trajs=4,run=0.h5
python scripts/imitate_mj.py --mode bclone --env HalfCheetah-v1 --data imitation_runs/modern_stochastic/trajs/trajs_halfcheetah.h5 --limit_trajs 11 --data_subsamp_freq 20 --max_iter 20001 --log imitation_runs/modern_stochastic/checkpoints_all/alg=bclone,task=halfcheetah,num_trajs=11,run=0.h5
python scripts/imitate_mj.py --mode bclone --env HalfCheetah-v1 --data imitation_runs/modern_stochastic/trajs/trajs_halfcheetah.h5 --limit_trajs 18 --data_subsamp_freq 20 --max_iter 20001 --log imitation_runs/modern_stochastic/checkpoints_all/alg=bclone,task=halfcheetah,num_trajs=18,run=0.h5
python scripts/imitate_mj.py --mode bclone --env HalfCheetah-v1 --data imitation_runs/modern_stochastic/trajs/trajs_halfcheetah.h5 --limit_trajs 25 --data_subsamp_freq 20 --max_iter 20001 --log imitation_runs/modern_stochastic/checkpoints_all/alg=bclone,task=halfcheetah,num_trajs=25,run=0.h5
python scripts/imitate_mj.py --mode ga --env Hopper-v1 --data imitation_runs/modern_stochastic/trajs/trajs_hopper.h5 --limit_trajs 4 --data_subsamp_freq 20 --favor_zero_expert_reward 0 --min_total_sa 50000 --max_iter 501 --reward_include_time 0 --reward_lr .01 --log imitation_runs/modern_stochastic/checkpoints_all/alg=ga,task=hopper,num_trajs=4,run=0.h5
python scripts/imitate_mj.py --mode ga --env Hopper-v1 --data imitation_runs/modern_stochastic/trajs/trajs_hopper.h5 --limit_trajs 11 --data_subsamp_freq 20 --favor_zero_expert_reward 0 --min_total_sa 50000 --max_iter 501 --reward_include_time 0 --reward_lr .01 --log imitation_runs/modern_stochastic/checkpoints_all/alg=ga,task=hopper,num_trajs=11,run=0.h5
python scripts/imitate_mj.py --mode ga --env Hopper-v1 --data imitation_runs/modern_stochastic/trajs/trajs_hopper.h5 --limit_trajs 18 --data_subsamp_freq 20 --favor_zero_expert_reward 0 --min_total_sa 50000 --max_iter 501 --reward_include_time 0 --reward_lr .01 --log imitation_runs/modern_stochastic/checkpoints_all/alg=ga,task=hopper,num_trajs=18,run=0.h5
python scripts/imitate_mj.py --mode ga --env Hopper-v1 --data imitation_runs/modern_stochastic/trajs/trajs_hopper.h5 --limit_trajs 25 --data_subsamp_freq 20 --favor_zero_expert_reward 0 --min_total_sa 50000 --max_iter 501 --reward_include_time 0 --reward_lr .01 --log imitation_runs/modern_stochastic/checkpoints_all/alg=ga,task=hopper,num_trajs=25,run=0.h5
python scripts/imitate_mj.py --mode ga --env Walker2d-v1 --data imitation_runs/modern_stochastic/trajs/trajs_walker.h5 --limit_trajs 4 --data_subsamp_freq 20 --favor_zero_expert_reward 0 --min_total_sa 50000 --max_iter 501 --reward_include_time 0 --reward_lr .01 --log imitation_runs/modern_stochastic/checkpoints_all/alg=ga,task=walker,num_trajs=4,run=0.h5
python scripts/imitate_mj.py --mode ga --env Walker2d-v1 --data imitation_runs/modern_stochastic/trajs/trajs_walker.h5 --limit_trajs 11 --data_subsamp_freq 20 --favor_zero_expert_reward 0 --min_total_sa 50000 --max_iter 501 --reward_include_time 0 --reward_lr .01 --log imitation_runs/modern_stochastic/checkpoints_all/alg=ga,task=walker,num_trajs=11,run=0.h5
python scripts/imitate_mj.py --mode ga --env Walker2d-v1 --data imitation_runs/modern_stochastic/trajs/trajs_walker.h5 --limit_trajs 18 --data_subsamp_freq 20 --favor_zero_expert_reward 0 --min_total_sa 50000 --max_iter 501 --reward_include_time 0 --reward_lr .01 --log imitation_runs/modern_stochastic/checkpoints_all/alg=ga,task=walker,num_trajs=18,run=0.h5
python scripts/imitate_mj.py --mode ga --env Walker2d-v1 --data imitation_runs/modern_stochastic/trajs/trajs_walker.h5 --limit_trajs 25 --data_subsamp_freq 20 --favor_zero_expert_reward 0 --min_total_sa 50000 --max_iter 501 --reward_include_time 0 --reward_lr .01 --log imitation_runs/modern_stochastic/checkpoints_all/alg=ga,task=walker,num_trajs=25,run=0.h5
python scripts/imitate_mj.py --mode ga --env Ant-v1 --data imitation_runs/modern_stochastic/trajs/trajs_ant.h5 --limit_trajs 4 --data_subsamp_freq 20 --favor_zero_expert_reward 0 --min_total_sa 50000 --max_iter 501 --reward_include_time 0 --reward_lr .01 --log imitation_runs/modern_stochastic/checkpoints_all/alg=ga,task=ant,num_trajs=4,run=0.h5
python scripts/imitate_mj.py --mode ga --env Ant-v1 --data imitation_runs/modern_stochastic/trajs/trajs_ant.h5 --limit_trajs 11 --data_subsamp_freq 20 --favor_zero_expert_reward 0 --min_total_sa 50000 --max_iter 501 --reward_include_time 0 --reward_lr .01 --log imitation_runs/modern_stochastic/checkpoints_all/alg=ga,task=ant,num_trajs=11,run=0.h5
python scripts/imitate_mj.py --mode ga --env Ant-v1 --data imitation_runs/modern_stochastic/trajs/trajs_ant.h5 --limit_trajs 18 --data_subsamp_freq 20 --favor_zero_expert_reward 0 --min_total_sa 50000 --max_iter 501 --reward_include_time 0 --reward_lr .01 --log imitation_runs/modern_stochastic/checkpoints_all/alg=ga,task=ant,num_trajs=18,run=0.h5
python scripts/imitate_mj.py --mode ga --env Ant-v1 --data imitation_runs/modern_stochastic/trajs/trajs_ant.h5 --limit_trajs 25 --data_subsamp_freq 20 --favor_zero_expert_reward 0 --min_total_sa 50000 --max_iter 501 --reward_include_time 0 --reward_lr .01 --log imitation_runs/modern_stochastic/checkpoints_all/alg=ga,task=ant,num_trajs=25,run=0.h5
python scripts/imitate_mj.py --mode ga --env HalfCheetah-v1 --data imitation_runs/modern_stochastic/trajs/trajs_halfcheetah.h5 --limit_trajs 4 --data_subsamp_freq 20 --favor_zero_expert_reward 0 --min_total_sa 50000 --max_iter 501 --reward_include_time 0 --reward_lr .01 --log imitation_runs/modern_stochastic/checkpoints_all/alg=ga,task=halfcheetah,num_trajs=4,run=0.h5
python scripts/imitate_mj.py --mode ga --env HalfCheetah-v1 --data imitation_runs/modern_stochastic/trajs/trajs_halfcheetah.h5 --limit_trajs 11 --data_subsamp_freq 20 --favor_zero_expert_reward 0 --min_total_sa 50000 --max_iter 501 --reward_include_time 0 --reward_lr .01 --log imitation_runs/modern_stochastic/checkpoints_all/alg=ga,task=halfcheetah,num_trajs=11,run=0.h5
python scripts/imitate_mj.py --mode ga --env HalfCheetah-v1 --data imitation_runs/modern_stochastic/trajs/trajs_halfcheetah.h5 --limit_trajs 18 --data_subsamp_freq 20 --favor_zero_expert_reward 0 --min_total_sa 50000 --max_iter 501 --reward_include_time 0 --reward_lr .01 --log imitation_runs/modern_stochastic/checkpoints_all/alg=ga,task=halfcheetah,num_trajs=18,run=0.h5
python scripts/imitate_mj.py --mode ga --env HalfCheetah-v1 --data imitation_runs/modern_stochastic/trajs/trajs_halfcheetah.h5 --limit_trajs 25 --data_subsamp_freq 20 --favor_zero_expert_reward 0 --min_total_sa 50000 --max_iter 501 --reward_include_time 0 --reward_lr .01 --log imitation_runs/modern_stochastic/checkpoints_all/alg=ga,task=halfcheetah,num_trajs=25,run=0.h5
python scripts/imitate_mj.py --mode ga --env Hopper-v1 --data imitation_runs/modern_stochastic/trajs/trajs_hopper.h5 --limit_trajs 4 --data_subsamp_freq 20 --favor_zero_expert_reward 0 --min_total_sa 50000 --max_iter 501 --reward_type l2ball --reward_include_time 0 --log imitation_runs/modern_stochastic/checkpoints_all/alg=fem,task=hopper,num_trajs=4,run=0.h5
python scripts/imitate_mj.py --mode ga --env Hopper-v1 --data imitation_runs/modern_stochastic/trajs/trajs_hopper.h5 --limit_trajs 11 --data_subsamp_freq 20 --favor_zero_expert_reward 0 --min_total_sa 50000 --max_iter 501 --reward_type l2ball --reward_include_time 0 --log imitation_runs/modern_stochastic/checkpoints_all/alg=fem,task=hopper,num_trajs=11,run=0.h5
python scripts/imitate_mj.py --mode ga --env Hopper-v1 --data imitation_runs/modern_stochastic/trajs/trajs_hopper.h5 --limit_trajs 18 --data_subsamp_freq 20 --favor_zero_expert_reward 0 --min_total_sa 50000 --max_iter 501 --reward_type l2ball --reward_include_time 0 --log imitation_runs/modern_stochastic/checkpoints_all/alg=fem,task=hopper,num_trajs=18,run=0.h5
python scripts/imitate_mj.py --mode ga --env Hopper-v1 --data imitation_runs/modern_stochastic/trajs/trajs_hopper.h5 --limit_trajs 25 --data_subsamp_freq 20 --favor_zero_expert_reward 0 --min_total_sa 50000 --max_iter 501 --reward_type l2ball --reward_include_time 0 --log imitation_runs/modern_stochastic/checkpoints_all/alg=fem,task=hopper,num_trajs=25,run=0.h5
python scripts/imitate_mj.py --mode ga --env Walker2d-v1 --data imitation_runs/modern_stochastic/trajs/trajs_walker.h5 --limit_trajs 4 --data_subsamp_freq 20 --favor_zero_expert_reward 0 --min_total_sa 50000 --max_iter 501 --reward_type l2ball --reward_include_time 0 --log imitation_runs/modern_stochastic/checkpoints_all/alg=fem,task=walker,num_trajs=4,run=0.h5
python scripts/imitate_mj.py --mode ga --env Walker2d-v1 --data imitation_runs/modern_stochastic/trajs/trajs_walker.h5 --limit_trajs 11 --data_subsamp_freq 20 --favor_zero_expert_reward 0 --min_total_sa 50000 --max_iter 501 --reward_type l2ball --reward_include_time 0 --log imitation_runs/modern_stochastic/checkpoints_all/alg=fem,task=walker,num_trajs=11,run=0.h5
python scripts/imitate_mj.py --mode ga --env Walker2d-v1 --data imitation_runs/modern_stochastic/trajs/trajs_walker.h5 --limit_trajs 18 --data_subsamp_freq 20 --favor_zero_expert_reward 0 --min_total_sa 50000 --max_iter 501 --reward_type l2ball --reward_include_time 0 --log imitation_runs/modern_stochastic/checkpoints_all/alg=fem,task=walker,num_trajs=18,run=0.h5
python scripts/imitate_mj.py --mode ga --env Walker2d-v1 --data imitation_runs/modern_stochastic/trajs/trajs_walker.h5 --limit_trajs 25 --data_subsamp_freq 20 --favor_zero_expert_reward 0 --min_total_sa 50000 --max_iter 501 --reward_type l2ball --reward_include_time 0 --log imitation_runs/modern_stochastic/checkpoints_all/alg=fem,task=walker,num_trajs=25,run=0.h5
python scripts/imitate_mj.py --mode ga --env Ant-v1 --data imitation_runs/modern_stochastic/trajs/trajs_ant.h5 --limit_trajs 4 --data_subsamp_freq 20 --favor_zero_expert_reward 0 --min_total_sa 50000 --max_iter 501 --reward_type l2ball --reward_include_time 0 --log imitation_runs/modern_stochastic/checkpoints_all/alg=fem,task=ant,num_trajs=4,run=0.h5
python scripts/imitate_mj.py --mode ga --env Ant-v1 --data imitation_runs/modern_stochastic/trajs/trajs_ant.h5 --limit_trajs 11 --data_subsamp_freq 20 --favor_zero_expert_reward 0 --min_total_sa 50000 --max_iter 501 --reward_type l2ball --reward_include_time 0 --log imitation_runs/modern_stochastic/checkpoints_all/alg=fem,task=ant,num_trajs=11,run=0.h5
python scripts/imitate_mj.py --mode ga --env Ant-v1 --data imitation_runs/modern_stochastic/trajs/trajs_ant.h5 --limit_trajs 18 --data_subsamp_freq 20 --favor_zero_expert_reward 0 --min_total_sa 50000 --max_iter 501 --reward_type l2ball --reward_include_time 0 --log imitation_runs/modern_stochastic/checkpoints_all/alg=fem,task=ant,num_trajs=18,run=0.h5
python scripts/imitate_mj.py --mode ga --env Ant-v1 --data imitation_runs/modern_stochastic/trajs/trajs_ant.h5 --limit_trajs 25 --data_subsamp_freq 20 --favor_zero_expert_reward 0 --min_total_sa 50000 --max_iter 501 --reward_type l2ball --reward_include_time 0 --log imitation_runs/modern_stochastic/checkpoints_all/alg=fem,task=ant,num_trajs=25,run=0.h5
python scripts/imitate_mj.py --mode ga --env HalfCheetah-v1 --data imitation_runs/modern_stochastic/trajs/trajs_halfcheetah.h5 --limit_trajs 4 --data_subsamp_freq 20 --favor_zero_expert_reward 0 --min_total_sa 50000 --max_iter 501 --reward_type l2ball --reward_include_time 0 --log imitation_runs/modern_stochastic/checkpoints_all/alg=fem,task=halfcheetah,num_trajs=4,run=0.h5
python scripts/imitate_mj.py --mode ga --env HalfCheetah-v1 --data imitation_runs/modern_stochastic/trajs/trajs_halfcheetah.h5 --limit_trajs 11 --data_subsamp_freq 20 --favor_zero_expert_reward 0 --min_total_sa 50000 --max_iter 501 --reward_type l2ball --reward_include_time 0 --log imitation_runs/modern_stochastic/checkpoints_all/alg=fem,task=halfcheetah,num_trajs=11,run=0.h5
python scripts/imitate_mj.py --mode ga --env HalfCheetah-v1 --data imitation_runs/modern_stochastic/trajs/trajs_halfcheetah.h5 --limit_trajs 18 --data_subsamp_freq 20 --favor_zero_expert_reward 0 --min_total_sa 50000 --max_iter 501 --reward_type l2ball --reward_include_time 0 --log imitation_runs/modern_stochastic/checkpoints_all/alg=fem,task=halfcheetah,num_trajs=18,run=0.h5
python scripts/imitate_mj.py --mode ga --env HalfCheetah-v1 --data imitation_runs/modern_stochastic/trajs/trajs_halfcheetah.h5 --limit_trajs 25 --data_subsamp_freq 20 --favor_zero_expert_reward 0 --min_total_sa 50000 --max_iter 501 --reward_type l2ball --reward_include_time 0 --log imitation_runs/modern_stochastic/checkpoints_all/alg=fem,task=halfcheetah,num_trajs=25,run=0.h5
python scripts/imitate_mj.py --mode ga --env Hopper-v1 --data imitation_runs/modern_stochastic/trajs/trajs_hopper.h5 --limit_trajs 4 --data_subsamp_freq 20 --favor_zero_expert_reward 0 --min_total_sa 50000 --max_iter 501 --reward_type simplex --reward_include_time 0 --log imitation_runs/modern_stochastic/checkpoints_all/alg=simplex,task=hopper,num_trajs=4,run=0.h5
python scripts/imitate_mj.py --mode ga --env Hopper-v1 --data imitation_runs/modern_stochastic/trajs/trajs_hopper.h5 --limit_trajs 11 --data_subsamp_freq 20 --favor_zero_expert_reward 0 --min_total_sa 50000 --max_iter 501 --reward_type simplex --reward_include_time 0 --log imitation_runs/modern_stochastic/checkpoints_all/alg=simplex,task=hopper,num_trajs=11,run=0.h5
python scripts/imitate_mj.py --mode ga --env Hopper-v1 --data imitation_runs/modern_stochastic/trajs/trajs_hopper.h5 --limit_trajs 18 --data_subsamp_freq 20 --favor_zero_expert_reward 0 --min_total_sa 50000 --max_iter 501 --reward_type simplex --reward_include_time 0 --log imitation_runs/modern_stochastic/checkpoints_all/alg=simplex,task=hopper,num_trajs=18,run=0.h5
python scripts/imitate_mj.py --mode ga --env Hopper-v1 --data imitation_runs/modern_stochastic/trajs/trajs_hopper.h5 --limit_trajs 25 --data_subsamp_freq 20 --favor_zero_expert_reward 0 --min_total_sa 50000 --max_iter 501 --reward_type simplex --reward_include_time 0 --log imitation_runs/modern_stochastic/checkpoints_all/alg=simplex,task=hopper,num_trajs=25,run=0.h5
python scripts/imitate_mj.py --mode ga --env Walker2d-v1 --data imitation_runs/modern_stochastic/trajs/trajs_walker.h5 --limit_trajs 4 --data_subsamp_freq 20 --favor_zero_expert_reward 0 --min_total_sa 50000 --max_iter 501 --reward_type simplex --reward_include_time 0 --log imitation_runs/modern_stochastic/checkpoints_all/alg=simplex,task=walker,num_trajs=4,run=0.h5
python scripts/imitate_mj.py --mode ga --env Walker2d-v1 --data imitation_runs/modern_stochastic/trajs/trajs_walker.h5 --limit_trajs 11 --data_subsamp_freq 20 --favor_zero_expert_reward 0 --min_total_sa 50000 --max_iter 501 --reward_type simplex --reward_include_time 0 --log imitation_runs/modern_stochastic/checkpoints_all/alg=simplex,task=walker,num_trajs=11,run=0.h5
python scripts/imitate_mj.py --mode ga --env Walker2d-v1 --data imitation_runs/modern_stochastic/trajs/trajs_walker.h5 --limit_trajs 18 --data_subsamp_freq 20 --favor_zero_expert_reward 0 --min_total_sa 50000 --max_iter 501 --reward_type simplex --reward_include_time 0 --log imitation_runs/modern_stochastic/checkpoints_all/alg=simplex,task=walker,num_trajs=18,run=0.h5
python scripts/imitate_mj.py --mode ga --env Walker2d-v1 --data imitation_runs/modern_stochastic/trajs/trajs_walker.h5 --limit_trajs 25 --data_subsamp_freq 20 --favor_zero_expert_reward 0 --min_total_sa 50000 --max_iter 501 --reward_type simplex --reward_include_time 0 --log imitation_runs/modern_stochastic/checkpoints_all/alg=simplex,task=walker,num_trajs=25,run=0.h5
python scripts/imitate_mj.py --mode ga --env Ant-v1 --data imitation_runs/modern_stochastic/trajs/trajs_ant.h5 --limit_trajs 4 --data_subsamp_freq 20 --favor_zero_expert_reward 0 --min_total_sa 50000 --max_iter 501 --reward_type simplex --reward_include_time 0 --log imitation_runs/modern_stochastic/checkpoints_all/alg=simplex,task=ant,num_trajs=4,run=0.h5
python scripts/imitate_mj.py --mode ga --env Ant-v1 --data imitation_runs/modern_stochastic/trajs/trajs_ant.h5 --limit_trajs 11 --data_subsamp_freq 20 --favor_zero_expert_reward 0 --min_total_sa 50000 --max_iter 501 --reward_type simplex --reward_include_time 0 --log imitation_runs/modern_stochastic/checkpoints_all/alg=simplex,task=ant,num_trajs=11,run=0.h5
python scripts/imitate_mj.py --mode ga --env Ant-v1 --data imitation_runs/modern_stochastic/trajs/trajs_ant.h5 --limit_trajs 18 --data_subsamp_freq 20 --favor_zero_expert_reward 0 --min_total_sa 50000 --max_iter 501 --reward_type simplex --reward_include_time 0 --log imitation_runs/modern_stochastic/checkpoints_all/alg=simplex,task=ant,num_trajs=18,run=0.h5
python scripts/imitate_mj.py --mode ga --env Ant-v1 --data imitation_runs/modern_stochastic/trajs/trajs_ant.h5 --limit_trajs 25 --data_subsamp_freq 20 --favor_zero_expert_reward 0 --min_total_sa 50000 --max_iter 501 --reward_type simplex --reward_include_time 0 --log imitation_runs/modern_stochastic/checkpoints_all/alg=simplex,task=ant,num_trajs=25,run=0.h5
python scripts/imitate_mj.py --mode ga --env HalfCheetah-v1 --data imitation_runs/modern_stochastic/trajs/trajs_halfcheetah.h5 --limit_trajs 4 --data_subsamp_freq 20 --favor_zero_expert_reward 0 --min_total_sa 50000 --max_iter 501 --reward_type simplex --reward_include_time 0 --log imitation_runs/modern_stochastic/checkpoints_all/alg=simplex,task=halfcheetah,num_trajs=4,run=0.h5
python scripts/imitate_mj.py --mode ga --env HalfCheetah-v1 --data imitation_runs/modern_stochastic/trajs/trajs_halfcheetah.h5 --limit_trajs 11 --data_subsamp_freq 20 --favor_zero_expert_reward 0 --min_total_sa 50000 --max_iter 501 --reward_type simplex --reward_include_time 0 --log imitation_runs/modern_stochastic/checkpoints_all/alg=simplex,task=halfcheetah,num_trajs=11,run=0.h5
python scripts/imitate_mj.py --mode ga --env HalfCheetah-v1 --data imitation_runs/modern_stochastic/trajs/trajs_halfcheetah.h5 --limit_trajs 18 --data_subsamp_freq 20 --favor_zero_expert_reward 0 --min_total_sa 50000 --max_iter 501 --reward_type simplex --reward_include_time 0 --log imitation_runs/modern_stochastic/checkpoints_all/alg=simplex,task=halfcheetah,num_trajs=18,run=0.h5
python scripts/imitate_mj.py --mode ga --env HalfCheetah-v1 --data imitation_runs/modern_stochastic/trajs/trajs_halfcheetah.h5 --limit_trajs 25 --data_subsamp_freq 20 --favor_zero_expert_reward 0 --min_total_sa 50000 --max_iter 501 --reward_type simplex --reward_include_time 0 --log imitation_runs/modern_stochastic/checkpoints_all/alg=simplex,task=halfcheetah,num_trajs=25,run=0.h5
END
cmd=$(echo "$COMMANDS" | awk "NR == $PBS_ARRAYID")
echo $cmd

read -r -d '' OUTPUTFILES << END
logs_im_modern_2018-11-15_21:13:25/0001_alg=bclone,task=hopper,num_trajs=4,run=0.txt
logs_im_modern_2018-11-15_21:13:25/0002_alg=bclone,task=hopper,num_trajs=11,run=0.txt
logs_im_modern_2018-11-15_21:13:25/0003_alg=bclone,task=hopper,num_trajs=18,run=0.txt
logs_im_modern_2018-11-15_21:13:25/0004_alg=bclone,task=hopper,num_trajs=25,run=0.txt
logs_im_modern_2018-11-15_21:13:25/0005_alg=bclone,task=walker,num_trajs=4,run=0.txt
logs_im_modern_2018-11-15_21:13:25/0006_alg=bclone,task=walker,num_trajs=11,run=0.txt
logs_im_modern_2018-11-15_21:13:25/0007_alg=bclone,task=walker,num_trajs=18,run=0.txt
logs_im_modern_2018-11-15_21:13:25/0008_alg=bclone,task=walker,num_trajs=25,run=0.txt
logs_im_modern_2018-11-15_21:13:25/0009_alg=bclone,task=ant,num_trajs=4,run=0.txt
logs_im_modern_2018-11-15_21:13:25/0010_alg=bclone,task=ant,num_trajs=11,run=0.txt
logs_im_modern_2018-11-15_21:13:25/0011_alg=bclone,task=ant,num_trajs=18,run=0.txt
logs_im_modern_2018-11-15_21:13:25/0012_alg=bclone,task=ant,num_trajs=25,run=0.txt
logs_im_modern_2018-11-15_21:13:25/0013_alg=bclone,task=halfcheetah,num_trajs=4,run=0.txt
logs_im_modern_2018-11-15_21:13:25/0014_alg=bclone,task=halfcheetah,num_trajs=11,run=0.txt
logs_im_modern_2018-11-15_21:13:25/0015_alg=bclone,task=halfcheetah,num_trajs=18,run=0.txt
logs_im_modern_2018-11-15_21:13:25/0016_alg=bclone,task=halfcheetah,num_trajs=25,run=0.txt
logs_im_modern_2018-11-15_21:13:25/0017_alg=ga,task=hopper,num_trajs=4,run=0.txt
logs_im_modern_2018-11-15_21:13:25/0018_alg=ga,task=hopper,num_trajs=11,run=0.txt
logs_im_modern_2018-11-15_21:13:25/0019_alg=ga,task=hopper,num_trajs=18,run=0.txt
logs_im_modern_2018-11-15_21:13:25/0020_alg=ga,task=hopper,num_trajs=25,run=0.txt
logs_im_modern_2018-11-15_21:13:25/0021_alg=ga,task=walker,num_trajs=4,run=0.txt
logs_im_modern_2018-11-15_21:13:25/0022_alg=ga,task=walker,num_trajs=11,run=0.txt
logs_im_modern_2018-11-15_21:13:25/0023_alg=ga,task=walker,num_trajs=18,run=0.txt
logs_im_modern_2018-11-15_21:13:25/0024_alg=ga,task=walker,num_trajs=25,run=0.txt
logs_im_modern_2018-11-15_21:13:25/0025_alg=ga,task=ant,num_trajs=4,run=0.txt
logs_im_modern_2018-11-15_21:13:25/0026_alg=ga,task=ant,num_trajs=11,run=0.txt
logs_im_modern_2018-11-15_21:13:25/0027_alg=ga,task=ant,num_trajs=18,run=0.txt
logs_im_modern_2018-11-15_21:13:25/0028_alg=ga,task=ant,num_trajs=25,run=0.txt
logs_im_modern_2018-11-15_21:13:25/0029_alg=ga,task=halfcheetah,num_trajs=4,run=0.txt
logs_im_modern_2018-11-15_21:13:25/0030_alg=ga,task=halfcheetah,num_trajs=11,run=0.txt
logs_im_modern_2018-11-15_21:13:25/0031_alg=ga,task=halfcheetah,num_trajs=18,run=0.txt
logs_im_modern_2018-11-15_21:13:25/0032_alg=ga,task=halfcheetah,num_trajs=25,run=0.txt
logs_im_modern_2018-11-15_21:13:25/0033_alg=fem,task=hopper,num_trajs=4,run=0.txt
logs_im_modern_2018-11-15_21:13:25/0034_alg=fem,task=hopper,num_trajs=11,run=0.txt
logs_im_modern_2018-11-15_21:13:25/0035_alg=fem,task=hopper,num_trajs=18,run=0.txt
logs_im_modern_2018-11-15_21:13:25/0036_alg=fem,task=hopper,num_trajs=25,run=0.txt
logs_im_modern_2018-11-15_21:13:25/0037_alg=fem,task=walker,num_trajs=4,run=0.txt
logs_im_modern_2018-11-15_21:13:25/0038_alg=fem,task=walker,num_trajs=11,run=0.txt
logs_im_modern_2018-11-15_21:13:25/0039_alg=fem,task=walker,num_trajs=18,run=0.txt
logs_im_modern_2018-11-15_21:13:25/0040_alg=fem,task=walker,num_trajs=25,run=0.txt
logs_im_modern_2018-11-15_21:13:25/0041_alg=fem,task=ant,num_trajs=4,run=0.txt
logs_im_modern_2018-11-15_21:13:25/0042_alg=fem,task=ant,num_trajs=11,run=0.txt
logs_im_modern_2018-11-15_21:13:25/0043_alg=fem,task=ant,num_trajs=18,run=0.txt
logs_im_modern_2018-11-15_21:13:25/0044_alg=fem,task=ant,num_trajs=25,run=0.txt
logs_im_modern_2018-11-15_21:13:25/0045_alg=fem,task=halfcheetah,num_trajs=4,run=0.txt
logs_im_modern_2018-11-15_21:13:25/0046_alg=fem,task=halfcheetah,num_trajs=11,run=0.txt
logs_im_modern_2018-11-15_21:13:25/0047_alg=fem,task=halfcheetah,num_trajs=18,run=0.txt
logs_im_modern_2018-11-15_21:13:25/0048_alg=fem,task=halfcheetah,num_trajs=25,run=0.txt
logs_im_modern_2018-11-15_21:13:25/0049_alg=simplex,task=hopper,num_trajs=4,run=0.txt
logs_im_modern_2018-11-15_21:13:25/0050_alg=simplex,task=hopper,num_trajs=11,run=0.txt
logs_im_modern_2018-11-15_21:13:25/0051_alg=simplex,task=hopper,num_trajs=18,run=0.txt
logs_im_modern_2018-11-15_21:13:25/0052_alg=simplex,task=hopper,num_trajs=25,run=0.txt
logs_im_modern_2018-11-15_21:13:25/0053_alg=simplex,task=walker,num_trajs=4,run=0.txt
logs_im_modern_2018-11-15_21:13:25/0054_alg=simplex,task=walker,num_trajs=11,run=0.txt
logs_im_modern_2018-11-15_21:13:25/0055_alg=simplex,task=walker,num_trajs=18,run=0.txt
logs_im_modern_2018-11-15_21:13:25/0056_alg=simplex,task=walker,num_trajs=25,run=0.txt
logs_im_modern_2018-11-15_21:13:25/0057_alg=simplex,task=ant,num_trajs=4,run=0.txt
logs_im_modern_2018-11-15_21:13:25/0058_alg=simplex,task=ant,num_trajs=11,run=0.txt
logs_im_modern_2018-11-15_21:13:25/0059_alg=simplex,task=ant,num_trajs=18,run=0.txt
logs_im_modern_2018-11-15_21:13:25/0060_alg=simplex,task=ant,num_trajs=25,run=0.txt
logs_im_modern_2018-11-15_21:13:25/0061_alg=simplex,task=halfcheetah,num_trajs=4,run=0.txt
logs_im_modern_2018-11-15_21:13:25/0062_alg=simplex,task=halfcheetah,num_trajs=11,run=0.txt
logs_im_modern_2018-11-15_21:13:25/0063_alg=simplex,task=halfcheetah,num_trajs=18,run=0.txt
logs_im_modern_2018-11-15_21:13:25/0064_alg=simplex,task=halfcheetah,num_trajs=25,run=0.txt
END
outputfile=$PBS_O_WORKDIR/$(echo "$OUTPUTFILES" | awk "NR == $PBS_ARRAYID")
echo $outputfile
# Make sure output directory exists
mkdir -p "`dirname "$outputfile"`" 2>/dev/null

cd $PBS_O_WORKDIR

echo $cmd >$outputfile
eval $cmd >>$outputfile 2>&1
